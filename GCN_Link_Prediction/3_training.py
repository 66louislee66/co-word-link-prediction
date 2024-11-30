# Train the whole model
# * import module
import dgl
import torch
import tqdm
import pandas as pd
from dgl.dataloading import MultiLayerFullNeighborSampler,as_edge_prediction_sampler
import itertools
from dgl import save_graphs
import gc
gc.collect()
from utils import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix,f1_score, roc_curve, auc

# TODO Training model

# * Setting up the device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# * Construct heterogeneous graph hetero_graph
hetero_graph,word_e_word_count,doc_e_word_count,topic_e_word_count = build_hetero_graph_train()
print(hetero_graph)

# * Side sampling and data loading
n_hetero_features = 16 
neg_sample_count = 1
batch_size=81920

sampler = MultiLayerFullNeighborSampler(2)  # Sample all nodes in layer 2
sampler = as_edge_prediction_sampler(sampler, negative_sampler=dgl.dataloading.negative_sampler.Uniform(neg_sample_count))

hetero_graph.edges['co-occurrence'].data['train_mask'] = torch.zeros(word_e_word_count, dtype=torch.bool).bernoulli(1.0)
train_word_eids = hetero_graph.edges['co-occurrence'].data['train_mask'].nonzero(as_tuple=True)[0]
word_dataloader = dgl.dataloading.DataLoader(  # Sub-batch training
    hetero_graph, {'co-occurrence': train_word_eids}, sampler,device, batch_size=batch_size, shuffle=True
)

hetero_graph.edges['tf-idf'].data['train_mask'] = torch.zeros(doc_e_word_count, dtype=torch.bool).bernoulli(1.0)
train_doc_eids = hetero_graph.edges['tf-idf'].data['train_mask'].nonzero(as_tuple=True)[0]
doc_dataloader = dgl.dataloading.DataLoader(
    hetero_graph, {'tf-idf': train_doc_eids}, sampler,device, batch_size=batch_size, shuffle=True
)

# hetero_graph.edges['LDA'].data['train_mask'] = torch.zeros(topic_e_word_count, dtype=torch.bool).bernoulli(1.0)
# train_topic_eids = hetero_graph.edges['LDA'].data['train_mask'].nonzero(as_tuple=True)[0]
# topic_dataloader = dgl.dataloading.DataLoader(
#     hetero_graph, {'LDA': train_topic_eids}, sampler,device, batch_size=batch_size, shuffle=True
# ) #1

# * Model training super-involved single-epoch training
hidden_feat_dim = n_hetero_features 
out_feat_dim = n_hetero_features 

embed_layer = RelGraphEmbed(hetero_graph, hidden_feat_dim)
all_node_embed = embed_layer()
all_node_embed = {ntype: embed.to(device) for ntype, embed in all_node_embed.items()}

model = Model(hetero_graph, hidden_feat_dim, out_feat_dim)

all_params = itertools.chain(model.parameters(), embed_layer.parameters())  # Optimise all the parameters of the model, mainly the weight and the input embedding parameters.
optimizer = torch.optim.Adam(all_params, lr=0.01, weight_decay=0)

loss_func = MarginLoss()

model.to(device)

def train_etype_one_epoch(etype, spec_dataloader):  # single training session
    losses = []
    for input_nodes, pos_g, neg_g, blocks in tqdm.tqdm(spec_dataloader):  # input nodes is the set of all nodes in the sampled subgraph
        emb = extract_embed(all_node_embed, input_nodes,device)
        pos_score, neg_score = model(emb, pos_g, neg_g, blocks, etype)
        loss = loss_func(pos_score, neg_score)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('{:s} Epoch {:d} | Loss {:.4f}'.format(etype, epoch, sum(losses) / len(losses)))

# * Model multiple node training
for epoch in range(50):
    print("start epoch:", epoch)
    model.train()
    train_etype_one_epoch('co-occurrence', word_dataloader)
    train_etype_one_epoch('tf-idf', doc_dataloader)
    # train_etype_one_epoch('LDA', topic_dataloader) #1
    
# * Model saving and node embedding exporting
save_graphs("graph.bin", [hetero_graph])  # Figure data and model preservation
torch.save(model.state_dict(), "model.bin")

print("node_embed:", all_node_embed['word'][0])
print(len(all_node_embed['word']))
print(len(all_node_embed['doc']))
# print(len(all_node_embed['topic']))  #1

# * The results of the model prediction should end up using inference and saving the derived results
# ! Note that all_node_embed is passed in here, select 0, selecting 1 may result in a deadlock and eventually the program will not execute
inference_out = model.inference(hetero_graph, batch_size,device, num_workers=0, x = all_node_embed)
print(inference_out['word'].shape)
print(inference_out['word'][0])

embed_matrix = inference_out['word'].cpu().detach().numpy()
embed_df = pd.DataFrame(embed_matrix)
embed_df.to_csv(r'./data/3_word_embeddings_inference.csv', index=False)

# TODO Testing model

# * Extract and combine positive and negative samples and save
comatix_test_label = pd.read_csv(r'./data/2_comatrix_test_label.csv', index_col=0)

stacked_comatrix = comatix_test_label.stack().reset_index()  
stacked_comatrix.columns = ['row_word', 'col_word', 'label']
upper_tri = stacked_comatrix[stacked_comatrix['row_word'] < stacked_comatrix['col_word']] 

positive_samples = upper_tri[upper_tri['label'] == 1]
negative_samples = upper_tri[upper_tri['label'] == 0]

balanced_data = pd.concat([positive_samples, negative_samples]) 
balanced_data.to_csv(r'./data/3_source_data_label.csv', index=False)

# * Convert names of positive and negative samples to indexes
wordid_decode_df = pd.read_csv('./data/3_wordid_decode_map.csv')
word_to_index = dict(zip(wordid_decode_df['Original Word'], wordid_decode_df['Encoded ID']))

balanced_data.loc[:, 'row_word'] = balanced_data['row_word'].map(word_to_index) 
balanced_data.loc[:, 'col_word'] = balanced_data['col_word'].map(word_to_index) 

# * Calculate the dot product and get the score, and save
pred_scores = []
dot_products = []

for idx, row in balanced_data.iterrows():
    row_index = row['row_word']
    col_index = row['col_word']
    row_embedding = inference_out['word'][row_index]
    col_embedding = inference_out['word'][col_index]
    dot_product = (row_embedding * col_embedding).sum().item()
    dot_products.append(dot_product)

dot_products_tensor = torch.tensor(dot_products)

mean = dot_products_tensor.mean().item() 
std = dot_products_tensor.std().item()

normalized_scores = [] 

for dot_product in dot_products: 
    normalized_score = (dot_product - mean) / std
    normalized_scores.append(normalized_score)

pred_scores = torch.sigmoid(torch.tensor(normalized_scores)) 

balanced_data['pred'] = pred_scores
balanced_data.to_csv(r'./data/3_modified_data_label.csv', index=False)

true_labels = balanced_data['label'] 
predicted_probs = balanced_data['pred']

predicted_labels = (predicted_probs > 0.5).astype(int)  

# * Evaluate the performance of the model and calculate the accuracy, precision and recall, confusion matrix and ROC
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)
conf_matrix = confusion_matrix(true_labels, predicted_labels)
fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs)
roc_auc = auc(fpr, tpr)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"ROC AUC: {roc_auc}")

# # Assuming upper_tri is a DataFrame you already have
# upper_tri = pd.read_csv(r'./data/3_modified_data_label.csv')
# thresholds = np.arange(0.3, 0.9, 0.01)
# best_f1 = 0
# best_threshold = 0

# for threshold in thresholds:
#     predicted_labels = (upper_tri['pred'] > threshold).astype(int)
#     accuracy = accuracy_score(upper_tri['label'], predicted_labels)
#     precision = precision_score(upper_tri['label'], predicted_labels)
#     recall = recall_score(upper_tri['label'], predicted_labels)
#     f1 = f1_score(upper_tri['label'], predicted_labels)
#     if f1 > best_f1:
#         best_f1 = f1
#         best_threshold = threshold

#     print(f"Threshold: {threshold}, accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")

# print(f"Best F1: {best_f1} at Threshold: {best_threshold}")