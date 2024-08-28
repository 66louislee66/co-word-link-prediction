import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix,f1_score, roc_curve, auc
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
import torch.nn.functional as F
from utils import *

# * 设置设备
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# * 加载节点数据(LDA特征)
df_LDA = pd.read_csv(r'./data/4_webofsci_train_LDA_features.csv', index_col = 0)
df_LDA = df_LDA.astype(np.float32)
LDA_arr = df_LDA.values.T
LDA_tensor = torch.from_numpy(LDA_arr)
LDA_tensor = LDA_tensor.to(device)

# * 训练集添加图数据
final_source_data_comatrix = pd.read_csv(r'./data/3_final_source_data_comatrix_train.csv')
reverse_edges = final_source_data_comatrix[['wordid2_encoded', 'wordid1_encoded', 'weight']].copy()
reverse_edges.columns = ['wordid1_encoded', 'wordid2_encoded', 'weight']
bidirectional_edges = pd.concat([final_source_data_comatrix, reverse_edges], ignore_index=True)
bidirectional_edges = bidirectional_edges.sort_values(by='wordid1_encoded', ascending=True)

data_train = data_create(bidirectional_edges,device,LDA_tensor)

# * 测试集添加图数据
comatix_test_label = pd.read_csv(r'./data/2_comatrix_test_label.csv', index_col=0)
stacked_comatrix = comatix_test_label.stack().reset_index()
stacked_comatrix.columns = ['row_word', 'col_word', 'label']

upper_tri = stacked_comatrix[stacked_comatrix['row_word'] < stacked_comatrix['col_word']]

balanced_data = upper_tri[upper_tri['label'] == 1] 

wordid_decode_df = pd.read_csv(r'./data/3_wordid_decode_map.csv')
word_to_index = dict(zip(wordid_decode_df['Original Word'], wordid_decode_df['Encoded ID']))

balanced_data.loc[:, 'row_word'] = balanced_data['row_word'].map(word_to_index)
balanced_data.loc[:, 'col_word'] = balanced_data['col_word'].map(word_to_index) 

reverse_edges_test = balanced_data[['col_word', 'row_word', 'label']].copy()
reverse_edges_test.columns = ['row_word', 'col_word', 'label']
bidirectional_edges_test = pd.concat([balanced_data, reverse_edges_test], ignore_index=True)
bidirectional_edges_test = bidirectional_edges_test.sort_values(by='row_word', ascending=True)

data_test = data_create(bidirectional_edges = bidirectional_edges_test, device = device,word_tensor = LDA_tensor)

class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).sigmoid()
        return self.conv2(x, edge_index)

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim = -1)
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

def get_link_labels(pos_edge_index, neg_edge_index, device):
    num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(num_links, dtype = torch.float)
    link_labels[:pos_edge_index.size(1)] = 1.
    link_labels = link_labels.to(device)
    return link_labels

def train(data, model, optimizer, device):
    model.train()
    
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.edge_index.size(1))
    optimizer.zero_grad()
    z = model.encode(data.x, data.edge_index)
    link_logits = model.decode(z, data.edge_index, neg_edge_index)
    link_labels = get_link_labels(data.edge_index, neg_edge_index, device)
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    loss.backward()
    optimizer.step()

    return loss

def test(data, model, device):
    model.eval()
    
    z = model.encode(data.x, data.edge_index)
    
    pos_edge_index = data.edge_index
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.edge_index.size(1))
    link_logits = model.decode(z, pos_edge_index, neg_edge_index)
    link_logits_np = link_logits.detach().cpu().numpy()
    mean = link_logits_np.mean()
    std = link_logits_np.std()
    link_logits_normalized = (link_logits_np - mean) / std
    link_probs = 1 / (1 + np.exp(-link_logits_normalized))
    # link_probs =link_logits.sigmoid()
    # link_probs = link_probs.detach().cpu().numpy()
    link_labels = get_link_labels(pos_edge_index, neg_edge_index, device)
    p = (link_probs > 0.5).astype('int64')
    y = link_labels.detach().cpu().numpy()
    accuracy = accuracy_score(y, p)
    precision = precision_score(y, p)
    recall = recall_score(y, p)
    f1 = f1_score(y, p)
    conf_matrix = confusion_matrix(y, p)
    fpr, tpr, thresholds = roc_curve(y, link_probs)
    roc_auc = auc(fpr, tpr)

    # 创建一个 DataFrame
    df = pd.DataFrame({
        'link_logits': link_logits_np,
        'link_probs': link_probs,
        'link_labels': y,
        'predictions': p
    })

    # 保存到 CSV 文件
    df.to_csv(r'./test/link_predictions.csv', index=False)
    
    
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"ROC AUC: {roc_auc}")
    
model = Net(data_train.x.shape[1], 8).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.1)

for epoch in range(1, 201):
    loss = train(data_train, model, optimizer, device)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

test(data_test, model, device)
z = model.encode(data_test.x, data_test.edge_index)
final_edge_index = model.decode_all(z)

