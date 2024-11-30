# custom function
# * import module
import numpy as np
import pandas as pd
import re
import dgl
import torch
import tqdm
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
import dgl.function as fn
from dgl.nn.pytorch import GraphConv, HeteroGraphConv
from dgl.nn import GraphConv
from sklearn.metrics import check_scoring
import torch.nn.functional as F
import numbers
from torch_geometric.data import Data
import csv
import gc
gc.collect()

# TODO 1_Webofsci_File_clean.py
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`-]", " ", string) 
    string = re.sub(r"\'s", " \'s", string)  
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string)  
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string)  
    string = re.sub(r"!", " ! ", string)  
    string = re.sub(r"\(", " \( ", string)  
    string = re.sub(r"\)", " \) ", string)  
    string = re.sub(r"\?", " \? ", string)  
    string = re.sub(r"\s{2,}", " ", string) 
    return string.strip().lower()  

# * Abandonment of special symbols
def is_valid(word):
    if re.match(r"[()\:;,.'-0-9]+", word):
        return False
    elif len(word) < 3:
        return False
    else:
        return True

# TODO 2_build_graph.py
def find_words(train_path, test_path):
    with open(train_path, 'r') as file:
        train_vocabulary = set(file.read().splitlines())
    unique_words = []
    with open(test_path, 'r') as file:
        for word in file:
            word = word.strip() 
            if word not in train_vocabulary:
                unique_words.append(word)
    return unique_words

# TODO 3_training.py
# * Unified Node Type
def encode_map(input_array):  # Encoding method
    p_map={}
    length=len(input_array)
    for index, ele in zip(range(length),input_array):
        p_map[str(ele)] = index
    return p_map

def decode_map(encode_map):  # Decoding method
    de_map={}
    for k,v in encode_map.items():
        de_map[v]=k
    return de_map

# * Constructing a heterogeneous graph
def build_hetero_graph_train(): 
    
    # Encoding map
    source_data_comatrix = pd.read_csv(r'./data/2_source_data_comatrix_train.csv')
    source_data_tfidf = pd.read_csv(r'./data/2_source_data_tfidf_train.csv')
    source_data_LDA = pd.read_csv(r'./data/2_source_data_LDA_train.csv') #1
    wordid_encode_map = encode_map(set(source_data_comatrix['wordid1'].values))
    docid_encode_map = encode_map(set(source_data_tfidf['docid'].values))
    topicid_encode_map = encode_map(set(source_data_LDA['topicid'].values)) #1
    
    # Decoding map 
    wordid_decode_map = decode_map(wordid_encode_map)
    source_data_comatrix['wordid1_encoded'] = source_data_comatrix['wordid1'].apply(lambda e: wordid_encode_map.get(str(e),-1))
    source_data_comatrix['wordid2_encoded'] = source_data_comatrix['wordid2'].apply(lambda e: wordid_encode_map.get(str(e),-1))
    docid_decode_map = decode_map(docid_encode_map)
    source_data_tfidf['docid_encoded'] = source_data_tfidf['docid'].apply(lambda e: docid_encode_map.get(str(e),-1))
    source_data_tfidf['wordid3_encoded'] = source_data_tfidf['wordid3'].apply(lambda e: wordid_encode_map.get(str(e),-1))
    topicid_decode_map = decode_map(topicid_encode_map) #1
    source_data_LDA['topicid_encoded'] = source_data_LDA['topicid'].apply(lambda e: topicid_encode_map.get(str(e),-1))
    source_data_LDA['wordid4_encoded'] = source_data_LDA['wordid4'].apply(lambda e: wordid_encode_map.get(str(e),-1))
    
    # Index and Word Correspondence
    with open('./data/3_wordid_decode_map.csv', mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Encoded ID', 'Original Word'])   
        for encoded_id, original_word in wordid_decode_map.items():
            writer.writerow([encoded_id, original_word])

    with open('./data/3_docid_decode_map.csv', mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Encoded ID', 'Original Doc ID'])
        for encoded_id, original_doc_id in docid_decode_map.items():
            writer.writerow([encoded_id, original_doc_id])
            
    # 1
    with open('./data/3_topicid_decode_map.csv', mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Encoded ID', 'Original Topic ID'])
        for encoded_id, original_topic_id in topicid_decode_map.items():
            writer.writerow([encoded_id, original_topic_id])
    
    # Counting the number of unique values 
    wordid1_count = len(set(source_data_comatrix['wordid1_encoded'].values))
    print(wordid1_count)
    wordid2_count = len(set(source_data_comatrix['wordid2_encoded'].values))
    print(wordid2_count)
    docid_count = len(set(source_data_tfidf['docid_encoded'].values))
    print(docid_count)
    wordid3_count = len(set(source_data_tfidf['wordid3_encoded'].values))
    print(wordid3_count)
    topicid_count = len(set(source_data_LDA['topicid_encoded'].values))  #1
    print(topicid_count)
    wordid4_count = len(set(source_data_LDA['wordid4_encoded'].values))
    print(wordid4_count)
    
    # Save the encoded matrix
    final_source_data_comatrix = source_data_comatrix[['wordid1_encoded','wordid2_encoded','weight']].sort_values(by='wordid1_encoded', ascending=True)
    final_source_data_comatrix.to_csv(r'./data/3_final_source_data_comatrix_train.csv',index=False)
    final_source_data_tfidf = source_data_tfidf[['docid_encoded','wordid3_encoded','weight']].sort_values(by='docid_encoded', ascending=True)
    final_source_data_tfidf.to_csv(r'./data/3_final_source_data_tfidf_train.csv',index=False)
    final_source_data_LDA = source_data_LDA[['topicid_encoded','wordid4_encoded','weight']].sort_values(by='topicid_encoded', ascending=True)
    final_source_data_LDA.to_csv(r'./data/3_final_source_data_LDA_train.csv',index=False) #1
    
    # word -co-occurence- word
    word_e_word_src = final_source_data_comatrix['wordid1_encoded'].values
    word_e_word_dst = final_source_data_comatrix['wordid2_encoded'].values
    co_occurrence_weights = final_source_data_comatrix['weight'].values
    word_e_word_count = len(word_e_word_dst)
    print("word_e_word_count", word_e_word_count)
    
    # doc -tfidf- word
    doc_e_word_src = final_source_data_tfidf['docid_encoded'].values
    doc_e_word_dst = final_source_data_tfidf['wordid3_encoded'].values
    tfidf_weights = final_source_data_tfidf['weight'].values
    doc_e_word_count = len(doc_e_word_dst)
    print("doc_e_word_count", doc_e_word_count)
    
    #1 topic -LDA- word  
    topic_e_word_src = final_source_data_LDA['topicid_encoded'].values
    topic_e_word_dst = final_source_data_LDA['wordid4_encoded'].values
    LDA_weights = final_source_data_LDA['weight'].values
    topic_e_word_count = len(topic_e_word_dst)
    print("topic_e_word_count", topic_e_word_count)

    graph_data = {
        ('word', 'co-occurrence', 'word'): (word_e_word_src, word_e_word_dst),
        ('word', 'co-occurrence_i', 'word'): (word_e_word_dst, word_e_word_src),
        ('doc', 'tf-idf', 'word'): (doc_e_word_src, doc_e_word_dst),
        ('word', 'tf-idf_i', 'doc'): (doc_e_word_dst, doc_e_word_src),
        # ('topic', 'LDA', 'word'): (topic_e_word_src, topic_e_word_dst),  #1
        # ('word', 'LDA_i', 'topic'): (topic_e_word_dst, topic_e_word_src) #1
    }
    
    g = dgl.heterograph(graph_data)
    
    # Setting edge features
    g.edges['co-occurrence'].data['weight'] = torch.tensor(co_occurrence_weights, dtype=torch.float32)
    g.edges['tf-idf'].data['weight'] = torch.tensor(tfidf_weights, dtype=torch.float32)
    # g.edges['LDA'].data['weight'] = torch.tensor(LDA_weights, dtype=torch.float32) #1
    
    return g, word_e_word_count, doc_e_word_count, topic_e_word_count

# * our method: heterogeneous graph model
class RelGraphConvLayer(nn.Module):
    def __init__(self,
                in_feat,
                out_feat,
                rel_names,
                num_bases, 
                 *, 
                weight=True,
                bias=True,
                activation=None,
                self_loop=False,
                dropout=0.0):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        self.conv = HeteroGraphConv({
            rel: GraphConv(in_feat, out_feat, norm='right', weight=False, bias=False)
            for rel in rel_names
        })

        self.use_weight = weight
        self.use_basis = num_bases < len(self.rel_names) and weight
        if self.use_weight:
            if self.use_basis:
                self.basis = dglnn.WeightBasis((in_feat, out_feat), num_bases, len(self.rel_names))  # 共享基础权重
            else:
                self.weight = nn.Parameter(torch.Tensor(len(self.rel_names), in_feat, out_feat))
                nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        # bias
        if bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)  

        # weight for self loop
        if self.self_loop: 
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):
        
        g = g.local_var() 
        if self.use_weight:
            weight = self.basis() if self.use_basis else self.weight
            wdict = {self.rel_names[i]: {'weight': w.squeeze(0)}  
                    for i, w in enumerate(torch.split(weight, 1, dim=0))} 
        else:
            wdict = {}

        if g.is_block: 
            inputs_src = inputs
            inputs_dst = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}
        else:
            inputs_src = inputs_dst = inputs

        hs = self.conv(g, inputs, mod_kwargs=wdict)
        
        # Feature transformations applied to each node type on a GCN
        def _apply(ntype, h):
            if self.self_loop:
                h = h + torch.matmul(inputs_dst[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)

        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}

# * Embedding layer for featureless heterogeneous graphs that does not depend on the initial features of the nodes
class RelGraphEmbed(nn.Module):
    r"""Embedding layer for featureless heterograph."""
    # 构造函数
    def __init__(self,
                g,
                embed_size,
                embed_name='embed',
                activation=None,
                dropout=0.0):
        super(RelGraphEmbed, self).__init__()
        self.g = g
        self.embed_size = embed_size
        self.embed_name = embed_name
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        self.embeds = nn.ParameterDict() 
        for ntype in g.ntypes:
            embed = nn.Parameter(torch.Tensor(g.number_of_nodes(ntype), self.embed_size))
            nn.init.xavier_uniform_(embed, gain=nn.init.calculate_gain('relu'))
            self.embeds[ntype] = embed

    def forward(self, block=None):
        
        return self.embeds

# * Used to classify entities in a diagram
class EntityClassify(nn.Module):

    def __init__(self,
                g,
                h_dim, out_dim,  
                num_bases=-1,  
                num_hidden_layers=1,  
                dropout=0,
                use_self_loop=False):
        super(EntityClassify, self).__init__()
        self.g = g
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.rel_names = list(set(g.etypes))
        self.rel_names.sort()
        if num_bases < 0 or num_bases > len(self.rel_names):
            self.num_bases = len(self.rel_names)
        else:
            self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop

        self.embed_layer = RelGraphEmbed(g, self.h_dim)
        self.layers = nn.ModuleList()

        self.layers.append(RelGraphConvLayer(
            self.h_dim, self.h_dim, self.rel_names,
            self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
            dropout=self.dropout, weight=False))  

        self.layers.append(RelGraphConvLayer(
            self.h_dim, self.out_dim, self.rel_names,
            self.num_bases, activation=None,
            self_loop=self.use_self_loop))

    def forward(self, h=None, blocks=None):
        if h is None: 

            h = self.embed_layer()
        if blocks is None:  

            for layer in self.layers:
                h = layer(self.g, h)
        else:
            # minibatch training
            for layer, block in zip(self.layers, blocks):  
                h = layer(block, h)
        return h

    def inference(self, g, batch_size, device, num_workers=0, x=None):

        if x is None:
            x = self.embed_layer()

        for l, layer in enumerate(self.layers): 
            y = {
                k: torch.zeros(
                    g.number_of_nodes(k),
                    self.h_dim if l != len(self.layers) - 1 else self.out_dim)
                for k in g.ntypes}

            
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.DataLoader(   
                g,
                {k: torch.arange(g.number_of_nodes(k)) for k in g.ntypes},
                sampler,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=num_workers)
            
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                # print(input_nodes)
                block = blocks[0].to(device)
                        
                h = {k: x[k][input_nodes[k]].to(device) for k in input_nodes.keys()}
                h = layer(block, h)

                for k in h.keys():
                    y[k][output_nodes[k]] = h[k].cpu()

            x = y  
        return y  
    
# * Model Sampling Overparticipation Side Sampling
def extract_embed(node_embed, input_nodes,device):
    emb = {}
    for ntype, nid in input_nodes.items():

        nid = nid.to(device)
        nid = input_nodes[ntype]
        emb[ntype] = node_embed[ntype][nid]
    return emb

# * Definition of thea model structure and description of the loss function
class HeteroDotProductPredictor(nn.Module):

    def forward(self, graph, h, etype):
        # Update h outside of computation, save as globally available
        with graph.local_scope(): 
            graph.ndata['h'] = h  
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']
        
# * Define a graph neural network model for link prediction
class Model(nn.Module):

    def __init__(self, graph, hidden_feat_dim, out_feat_dim):
        super().__init__()
        self.Hetegcn = EntityClassify(graph,
                                hidden_feat_dim,
                                out_feat_dim)
        self.pred = HeteroDotProductPredictor()

    def forward(self, h, pos_g, neg_g, blocks, etype):
        h = self.Hetegcn(h, blocks)
        return self.pred(pos_g, h, etype), self.pred(neg_g, h, etype)  
    
    def inference(self, hetero_graph, batch_size, device, num_workers, x):
        return self.Hetegcn.inference(hetero_graph, batch_size, device, num_workers, x)
    
# * Customised loss function
class MarginLoss(nn.Module):

    def forward(self, pos_score, neg_score):
        return (1 - pos_score + neg_score.view(pos_score.shape[0], -1)).clamp(min=0).mean()    

# TODO 5_Comparative Experiment_traditional.py
# * Adding graph data to the training and test sets
def data_create(bidirectional_edges,device,word_tensor):
    edge_list_1 = []
    edge_list_2 = []

    for index, row in bidirectional_edges.iterrows():
        edge_list_1.append(int(row.iloc[0]))
        edge_list_2.append(int(row.iloc[1]))

    print(edge_list_1[:5])
    print(edge_list_2[:5])

    pos_num = len(edge_list_1)
    print('positive sample size: ',pos_num)

    arr = np.zeros((2,pos_num), dtype=np.int64)
    arr[0] = edge_list_1
    arr[1] = edge_list_2
    edge_index = torch.from_numpy(arr)
    edge_index = edge_index.to(device)

    data_wos = Data(x = word_tensor, edge_index = edge_index)    
    
    return data_wos

# * Scoring indicators
def get_scores(clf, X_new, y_new):
    scoring = ['precision', 'recall', 'accuracy', 'roc_auc', 'f1', 'average_precision']
    scorers = {scorer_name: check_scoring(clf, scorer_name) for scorer_name in scoring}
    #print(scorers)
    scores = multimetric_score(clf, X_new, y_new, scorers)
    return scores

def multimetric_score(estimator, X_test, y_test, scorers):
    """Return a dict of score for multimetric scoring"""
    scores = {}
    for name, scorer in scorers.items():
        if y_test is None:
            score = scorer(estimator, X_test)
        else:
            score = scorer(estimator, X_test, y_test)

        if hasattr(score, 'item'):
            try:
                # e.g. unwrap memmapped scalars
                score = score.item()
            except ValueError:
                # non-scalar?
                pass
        scores[name] = score

        if not isinstance(score, numbers.Number):
            raise ValueError("scoring must return a number, got %s (%s) "
                            "instead. (scorer=%s)"
                            % (str(score), type(score), name))
    return scores