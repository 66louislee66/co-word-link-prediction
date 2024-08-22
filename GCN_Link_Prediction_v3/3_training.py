# 训练整个模型
# * 导入模块
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh  # .eigen.arpack
import sys
import re
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import dgl
import torch
import tqdm
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
import dgl.function as fn
from dgl.nn.pytorch import GraphConv, SAGEConv, HeteroGraphConv
import argparse
import torch.optim as optim
from dgl.dataloading import MultiLayerFullNeighborSampler,DataLoader,as_edge_prediction_sampler
from dgl.dataloading.negative_sampler import Uniform
import itertools
import os
from dgl import save_graphs, load_graphs
from dgl.utils import expand_as_pair
from collections import defaultdict
from dgl.data.utils import makedirs, save_info, load_info
from sklearn.metrics import roc_auc_score
import gc
gc.collect()
from dgl.nn import GraphConv,SAGEConv
from utils import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix,f1_score, roc_curve, auc

# TODO Training model

# * 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# * 构建异构图hetero_graph
hetero_graph,word_e_word_count,doc_e_word_count = build_hetero_graph_train()
print(hetero_graph)

# * 边采样和数据加载
n_hetero_features = 16  # 特征维度大小
neg_sample_count = 1
batch_size=81920


sampler = MultiLayerFullNeighborSampler(2)  # 采样2层全部节点
sampler = as_edge_prediction_sampler(sampler, negative_sampler=dgl.dataloading.negative_sampler.Uniform(neg_sample_count))

hetero_graph.edges['co-occurrence'].data['train_mask'] = torch.zeros(word_e_word_count, dtype=torch.bool).bernoulli(1.0)
train_word_eids = hetero_graph.edges['co-occurrence'].data['train_mask'].nonzero(as_tuple=True)[0]
word_dataloader = dgl.dataloading.DataLoader(  # 分batch训练
    hetero_graph, {'co-occurrence': train_word_eids}, sampler,device, batch_size=batch_size, shuffle=True
)

hetero_graph.edges['tf-idf'].data['train_mask'] = torch.zeros(doc_e_word_count, dtype=torch.bool).bernoulli(1.0)
train_doc_eids = hetero_graph.edges['tf-idf'].data['train_mask'].nonzero(as_tuple=True)[0]
doc_dataloader = dgl.dataloading.DataLoader(
    hetero_graph, {'tf-idf': train_doc_eids}, sampler,device, batch_size=batch_size, shuffle=True
)

# ? 模型训练超参与单epoch训练
# in_feats = hetero_graph.nodes['user'].data['feature'].shape[1]
hidden_feat_dim = n_hetero_features
out_feat_dim = n_hetero_features

embed_layer = RelGraphEmbed(hetero_graph, hidden_feat_dim)
all_node_embed = embed_layer()
all_node_embed = {ntype: embed.to(device) for ntype, embed in all_node_embed.items()}
# all_node_embed = nn.ParameterDict()
# for ntype in hetero_graph.ntypes:
#     # 创建一个单位矩阵，每一行是一个节点的one-hot编码
#     embed = nn.Parameter(torch.eye(hetero_graph.number_of_nodes(ntype)))
#     all_node_embed[ntype] = embed

model = Model(hetero_graph, hidden_feat_dim, out_feat_dim)
# 优化模型所有参数,主要是weight以及输入的embeding参数
all_params = itertools.chain(model.parameters(), embed_layer.parameters())
optimizer = torch.optim.Adam(all_params, lr=0.01, weight_decay=0)

loss_func = MarginLoss()

# 将数据加载到 GPU
model.to(device)  # 将模型移动到 GPU

def train_etype_one_epoch(etype, spec_dataloader):
    losses = []
    # *input nodes为采样的subgraph中的所有的节点的集合
    for input_nodes, pos_g, neg_g, blocks in tqdm.tqdm(spec_dataloader):
        emb = extract_embed(all_node_embed, input_nodes,device)
        pos_score, neg_score = model(emb, pos_g, neg_g, blocks, etype)
        loss = loss_func(pos_score, neg_score)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('{:s} Epoch {:d} | Loss {:.4f}'.format(etype, epoch, sum(losses) / len(losses)))

# ? 模型多种节点训练
# *开始train 模型
for epoch in range(14):
    print("start epoch:", epoch)
    model.train()
    train_etype_one_epoch('co-occurrence', word_dataloader)
    train_etype_one_epoch('tf-idf', doc_dataloader)
    
# ? 模型保存与节点embedding导出
# * 图数据和模型保存
save_graphs("graph.bin", [hetero_graph])
torch.save(model.state_dict(), "model.bin")

# * 每个节点的embeding,自己初始化,因为参与了训练,这个就是最后每个节点输出的embedding
print("node_embed:", all_node_embed['word'][0])
print(len(all_node_embed['word']))
print(len(all_node_embed['doc']))

# embed_matrix = all_node_embed['word'].cpu().detach().numpy()
# embed_df = pd.DataFrame(embed_matrix)
# embed_df.to_csv(r'/home/lym/lab/project_work/project_versions/GCN_Link_Prediction_v3/data/word_embeddings.csv', index=False)

# * 模型预估的结果,最后应该使用 inference,这里得到的是logit
# * 注意,这里传入 all_node_embed,选择0,选1可能会死锁,最终程序不执行
inference_out = model.inference(hetero_graph, batch_size,device, num_workers=0, x = all_node_embed)
print(inference_out['word'].shape)
print(inference_out['word'][0])

embed_matrix = inference_out['word'].cpu().detach().numpy()
embed_df = pd.DataFrame(embed_matrix)
embed_df.to_csv(r'/home/lym/lab/project_work/project_versions/GCN_Link_Prediction_v3/data/word_embeddings_inference.csv', index=False)

# ! Testing model

comatix_test_label = pd.read_csv(r'/home/lym/lab/project_work/project_versions/GCN_Link_Prediction_v3/data/comatrix_test_label.csv', index_col=0)

# 将DataFrame转换为长格式，并保留原始的行和列标签
stacked_comatrix = comatix_test_label.stack().reset_index()

# 重命名列以更清晰地表示数据
stacked_comatrix.columns = ['row_word', 'col_word', 'label']

# 过滤出上三角部分的数据（不包括对角线）
upper_tri = stacked_comatrix[stacked_comatrix['row_word'] < stacked_comatrix['col_word']]

# 分离正样本和负样本
positive_samples = upper_tri[upper_tri['label'] == 1]
negative_samples = upper_tri[upper_tri['label'] == 0]

# # 随机选择与正样本数量相同的负样本
# balanced_negative_samples = negative_samples.sample(n=len(positive_samples), random_state=42)

# 合并正样本和选定的负样本
balanced_data = pd.concat([positive_samples, negative_samples])


balanced_data.to_csv(r'/home/lym/lab/project_work/project_versions/GCN_Link_Prediction_v3/data/source_data_label.csv', index=False)

# 加载 wordid_decode_map.csv 文件
wordid_decode_df = pd.read_csv('./data/wordid_decode_map.csv')

# 创建从词名称到索引的映射字典
word_to_index = dict(zip(wordid_decode_df['Original Word'], wordid_decode_df['Encoded ID']))

# 假设 balanced_data 是您已经有的DataFrame，其中包含词的名称
# 替换 'row_word' 和 'col_word' 列中的词名称为索引
balanced_data.loc[:, 'row_word'] = balanced_data['row_word'].map(word_to_index)
balanced_data.loc[:, 'col_word'] = balanced_data['col_word'].map(word_to_index) 

# 初始化一个空列表来存储点积结果
pred_scores = []
dot_products = []
# 遍历 balanced_data DataFrame
for idx, row in balanced_data.iterrows():
    # 获取两个词的索引
    row_index = row['row_word']
    col_index = row['col_word']
    
    # 从 inference_out 中获取对应的嵌入
    row_embedding = inference_out['word'][row_index]
    col_embedding = inference_out['word'][col_index]
    
    # 计算点积
    dot_product = (row_embedding * col_embedding).sum().item()
    dot_products.append(dot_product)
# 将点积列表转换为张量
dot_products_tensor = torch.tensor(dot_products)

# 计算平均值和标准差
mean = dot_products_tensor.mean().item()
std = dot_products_tensor.std().item()

# 初始化一个空列表来存储标准化后的得分
normalized_scores = []

# 对每个点积进行标准化
for dot_product in dot_products:
    normalized_score = (dot_product - mean) / std
    normalized_scores.append(normalized_score)

# 应用sigmoid函数将标准化得分转换为概率
pred_scores = torch.sigmoid(torch.tensor(normalized_scores))

# 将点积结果列表添加到 balanced_data DataFrame 的新列 "pred"
balanced_data['pred'] = pred_scores

balanced_data.to_csv(r'/home/lym/lab/project_work/project_versions/GCN_Link_Prediction_v3/data/modified_data_label.csv', index=False)

# 提取真实标签和预测概率
true_labels = balanced_data['label']
predicted_probs = balanced_data['pred']

# 将预测概率二值化（假设阈值为0.5）
predicted_labels = (predicted_probs > 0.5).astype(int)

# 计算准确率、精确率和召回率
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)

# 计算混淆矩阵
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# 计算ROC曲线和AUC
fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs)
roc_auc = auc(fpr, tpr)

# 打印结果
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"ROC AUC: {roc_auc}")

# # 假设 upper_tri 是您已经有的DataFrame
# upper_tri = pd.read_csv(r'/home/lym/lab/project_work/project_versions/GCN_Link_Prediction_v3/data/modified_data_label.csv')
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