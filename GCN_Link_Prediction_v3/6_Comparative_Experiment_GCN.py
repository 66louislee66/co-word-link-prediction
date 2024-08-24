import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
import torch.nn.functional as F
from utils import *

# * 设置设备
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# * 获取正负样本的索引并转到GPU上
df_comatrix = pd.read_csv(r'./data/2_comatrix_train.csv', index_col=0)
df_comatrix[df_comatrix != 0] = 1
df_comatrix.to_csv(r'./data/6_comatrix_train.csv')

vocab = df_comatrix.index.tolist()

edge_list_1 = []
edge_list_2 = []

for i in range(len(vocab)):
    print(i)
    for j in range(i + 1, len(vocab)):
        if df_comatrix.iloc[i, j] == 1:
            edge_list_1.append(vocab[i])
            edge_list_2.append(vocab[j])

temp = len(edge_list_1)
print(temp)

arr = np.zeros((2,len(edge_list_1 + edge_list_2)), dtype=np.int64)
arr[0] = [vocab.index(word) for word in edge_list_1] + [vocab.index(word) for word in edge_list_2]
arr[1] = [vocab.index(word) for word in edge_list_2] + [vocab.index(word) for word in edge_list_1]

edge_index = torch.from_numpy(arr)
edge_index = edge_index.to(device)

# * 读取特征矩阵并转到GPU上
df_LDA = pd.read_csv(r'./data/4_webofsci_train_LDA_features.csv', index_col = 0)
df_LDA = df_LDA.astype(np.float32)
LDA_arr = df_LDA.values.T
LDA_tensor = torch.from_numpy(LDA_arr)
LDA_tensor = LDA_tensor.to(device)
print(LDA_tensor.shape)

# * 将数据添加到data上
data_wos = Data(x = LDA_tensor, edge_index = edge_index)
data_wos = train_test_split_edges(data_wos, val_ratio = 0.1, test_ratio = 0.1)
class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 128)
        self.conv2 = GCNConv(128, out_channels)
        #self.conv1 = GCNConv(in_channels, 8)
        #self.conv2 = GCNConv(8, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim = -1)
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

def get_link_labels(pos_edge_index, neg_edge_index):
    num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(num_links, dtype = torch.float)
    link_labels[:pos_edge_index.size(1)] = 1.
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    link_labels = link_labels.to(device)
    return link_labels

def train(data, model, optimizer):
    model.train()
    
    neg_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1))
    optimizer.zero_grad()
    z = model.encode(data.x, data.train_pos_edge_index)
    link_logits = model.decode(z, data.train_pos_edge_index, neg_edge_index)
    link_labels = get_link_labels(data.train_pos_edge_index, neg_edge_index)
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    loss.backward()
    optimizer.step()

    return loss

def test(data, model):
    model.eval()
    
    z = model.encode(data.x, data.train_pos_edge_index)
    
    results = []
    for prefix in ['val', 'test']:
        pos_edge_index = data[f'{prefix}_pos_edge_index']
        neg_edge_index = data[f'{prefix}_neg_edge_index']
        link_logits = model.decode(z, pos_edge_index, neg_edge_index)
        link_probs = link_logits.sigmoid()
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        results.append(roc_auc_score(link_labels.detach().cpu().numpy(), link_probs.detach().cpu().numpy()))
        a = link_probs.detach().cpu().numpy()
        a = a + 0.5
        a = a.astype('int64')
        print("accuracy:", accuracy_score(link_labels.detach().cpu().numpy(), a))
        print("recall:", recall_score(link_labels.detach().cpu().numpy(), a))
        print("precision:", precision_score(link_labels.detach().cpu().numpy(), a))
    return results
data = data_wos.to(device)
data = data_wos
model = Net(data.x.shape[1], 8).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    
best_val_auc = test_auc = 0
for epoch in range(1, 201):
    loss = train(data, model, optimizer)
    val_auc, tmp_test_auc = test(data, model)
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        test_auc = tmp_test_auc
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, '
          f'Test: {test_auc:.4f}')
    
z = model.encode(data.x, data.train_pos_edge_index)
final_edge_index = model.decode_all(z)
    
