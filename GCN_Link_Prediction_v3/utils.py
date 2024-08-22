import numpy as np
import pandas as pd
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
from dgl.nn import GraphConv,SAGEConv
import argparse
import torch.optim as optim
from dgl.dataloading import MultiLayerFullNeighborSampler, GraphDataLoader, DataLoader
from dgl.dataloading.negative_sampler import Uniform
import itertools
import os
from dgl import save_graphs, load_graphs
from dgl.utils import expand_as_pair
from collections import defaultdict
from dgl.data.utils import makedirs, save_info, load_info
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.metrics import check_scoring
from sklearn.preprocessing import StandardScaler
import numbers
from torch_geometric.data import Data
import csv
import gc
gc.collect()

# ! Pubmed_File_clean.py
def loadWord2Vec(filename):  # 定义一个函数，名为 loadWord2Vec，它接受一个参数，即文件名
    """Read Word Vectors"""
    vocab = []  # 定义三个空列表，分别用于存储单词表、单词向量的列表和单词向量的字典
    embd = []
    word_vector_map = {}
    file = open(filename, 'r') 
    for line in file.readlines():  # 使用一个 for 循环，遍历文件中的每一行，每一行的内容赋值给变量 line
        row = line.strip().split(' ')  # 使用 strip 方法，去掉行首和行尾的空白字符，接着，使用 split 方法，以空格为分隔符，将字符串分割成一个列表，赋值给变量 row。
        if(len(row) > 2):  # 判断列表的长度是否大于 2，如果是，说明这一行是有效的单词向量数据
            vocab.append(row[0])  # 将列表的第一个元素，即单词，添加到 vocab 列表中
            vector = row[1:]  # 将列表的剩余元素，即单词向量，赋值给变量 vector
            length = len(vector)
            for i in range(length):  
                vector[i] = float(vector[i])  # 将单词向量的元素从字符串类型转换成浮点数类型，覆盖原来的值
            embd.append(vector)  # 将单词向量添加到 embd 列表中
            word_vector_map[row[0]] = vector  # 将单词和单词向量作为键值对，添加到 word_vector_map 字典中
    print('Loaded Word Vectors!')  
    file.close()
    return vocab, embd, word_vector_map

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`-]", " ", string) # 对字符串进行替换，使用re模块中的sub函数，第一个参数是一个正则表达式，表示匹配除了字母、数字、括号、逗号、感叹号、问号、单引号和反引号之外的任何字符，第二个参数是一个空格，表示将匹配到的字符替换为一个空格，第三个参数是字符串本身，表示在哪个字符串上进行替换，最后将替换后的字符串赋值给string变量
    string = re.sub(r"\'s", " \'s", string)  # 对字符串进行替换，使用re模块中的sub函数，第一个参数是一个正则表达式，表示匹配单引号后面跟着一个s的字符，第二个参数是一个空格加上单引号和s，表示将匹配到的字符替换为一个空格加上单引号和s，第三个参数是字符串本身，表示在哪个字符串上进行替换，最后将替换后的字符串赋值给string变量
    string = re.sub(r"\'ve", " \'ve", string)  # 对字符串进行替换，使用re模块中的sub函数，第一个参数是一个正则表达式，表示匹配单引号后面跟着一个ve的字符，第二个参数是一个空格加上单引号和ve，表示将匹配到的字符替换为一个空格加上单引号和ve，第三个参数是字符串本身，表示在哪个字符串上进行替换，最后将替换后的字符串赋值给string变量
    string = re.sub(r"n\'t", " n\'t", string)  # 对字符串进行替换，使用re模块中的sub函数，第一个参数是一个正则表达式，表示匹配一个n后面跟着一个单引号和t的字符，第二个参数是一个空格加上n、单引号和t，表示将匹配到的字符替换为一个空格加上n、单引号和t，第三个参数是字符串本身，表示在哪个字符串上进行替换，最后将替换后的字符串赋值给string变量
    string = re.sub(r"\'re", " \'re", string)  # 对字符串进行替换，使用re模块中的sub函数，第一个参数是一个正则表达式，表示匹配单引号后面跟着一个re的字符，第二个参数是一个空格加上单引号和re，表示将匹配到的字符替换为一个空格加上单引号和re，第三个参数是字符串本身，表示在哪个字符串上进行替换，最后将替换后的字符串赋值给string变量
    string = re.sub(r"\'d", " \'d", string)  # 对字符串进行替换，使用re模块中的sub函数，第一个参数是一个正则表达式，表示匹配单引号后面跟着一个d的字符，第二个参数是一个空格加上单引号和d，表示将匹配到的字符替换为一个空格加上单引号和d，第三个参数是字符串本身，表示在哪个字符串上进行替换，最后将替换后的字符串赋值给string变量
    string = re.sub(r"\'ll", " \'ll", string)  # 对字符串进行替换，使用re模块中的sub函数，第一个参数是一个正则表达式，表示匹配单引号后面跟着一个ll的字符，第二个参数是一个空格加上单引号和ll，表示将匹配到的字符替换为一个空格加上单引号和ll，第三个参数是字符串本身，表示在哪个字符串上进行替换，最后将替换后的字符串赋值给string变量
    string = re.sub(r",", " , ", string)  # 对字符串进行替换，使用re模块中的sub函数，第一个参数是一个正则表达式，表示匹配一个逗号的字符，第二个参数是一个空格加上逗号和空格，表示将匹配到的字符替换为一个空格加上逗号和空格，第三个参数是字符串本身，表示在哪个字符串上进行替换，最后将替换后的字符串赋值给string变量
    string = re.sub(r"!", " ! ", string)  # 对字符串进行替换，使用re模块中的sub函数，第一个参数是一个正则表达式，表示匹配一个感叹号的字符，第二个参数是一个空格加上感叹号和空格，表示将匹配到的字符替换为一个空格加上感叹号和空格，第三个参数是字符串本身，表示在哪个字符串上进行替换，最后将替换后的字符串赋值给string变量
    string = re.sub(r"\(", " \( ", string)  # 对字符串进行替换，使用re模块中的sub函数，第一个参数是一个正则表达式，表示匹配一个左括号的字符，第二个参数是一个空格加上左括号和空格，表示将匹配到的字符替换为一个空格加上左括号和空格，第三个参数是字符串本身，表示在哪个字符串上进行替换，最后将替换后的字符串赋值给string变量
    string = re.sub(r"\)", " \) ", string)  # 对字符串进行替换，使用re模块中的sub函数，第一个参数是一个正则表达式，表示匹配一个右括号的字符，第二个参数是一个空格加上右括号和空格，表示将匹配到的字符替换为一个空格加上右括号和空格，第三个参数是字符串本身，表示在哪个字符串上进行替换，最后将替换后的字符串赋值给string变量
    string = re.sub(r"\?", " \? ", string)  # 对字符串进行替换，使用re模块中的sub函数，第一个参数是一个正则表达式，表示匹配一个问号的字符，第二个参数是一个空格加上问号和空格，表示将匹配到的字符替换为一个空格加上问号和空格，第三个参数是字符串本身，表示在哪个字符串上进行替换，最后将替换后的字符串赋值给string变量
    string = re.sub(r"\s{2,}", " ", string) # 对字符串进行替换，使用re模块中的sub函数，第一个参数是一个正则表达式，表示匹配两个或以上的空白字符，包括空格、制表符、换行符等，第二个参数是一个空格，表示将匹配到的字符替换为一个空格，第三个参数是字符串本身，表示在哪个字符串上进行替换，最后将替换后的字符串赋值给string变量
    return string.strip().lower()  # 返回一个处理后的字符串，使用字符串对象的strip和lower方法，strip方法的作用是去掉字符串两端的空白字符，lower方法的作用是将字符串中的所有大写字母转换为小写字母，这样可以统一字符串的格式

# ?获取单词的词性
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None
    
# ? 预处理,判断是否存在特殊符号,如果是False就丢掉
def is_valid(word):
    if re.match(r"[()\:;,.'-0-9]+", word):
        return False
    elif len(word) < 3:
        return False
    else:
        return True

# ! training.py
# * 统一节点类型
def encode_map(input_array):  # 编码方法
    p_map={}
    length=len(input_array)
    for index, ele in zip(range(length),input_array):
        p_map[str(ele)] = index
    return p_map

def decode_map(encode_map):  # 解码方法
    de_map={}
    for k,v in encode_map.items():
        de_map[v]=k
    return de_map

# * 构建训练集的异构图
def build_hetero_graph_train():  # wordid1、wordid2、docid、wordid3编码解码
    
    # 编码map
    source_data_comatrix = pd.read_csv(r'/home/lym/lab/project_work/project_versions/GCN_Link_Prediction_v3/data/source_data_comatrix_train.csv')
    source_data_tfidf = pd.read_csv(r'/home/lym/lab/project_work/project_versions/GCN_Link_Prediction_v3/data/source_data_tfidf_train.csv')
    wordid_encode_map = encode_map(set(source_data_comatrix['wordid1'].values))
    docid_encode_map = encode_map(set(source_data_tfidf['docid'].values))
    
    # 解码map 
    wordid_decode_map = decode_map(wordid_encode_map)
    source_data_comatrix['wordid1_encoded'] = source_data_comatrix['wordid1'].apply(lambda e: wordid_encode_map.get(str(e),-1))
    source_data_comatrix['wordid2_encoded'] = source_data_comatrix['wordid2'].apply(lambda e: wordid_encode_map.get(str(e),-1))
    docid_decode_map = decode_map(docid_encode_map)
    source_data_tfidf['docid_encoded'] = source_data_tfidf['docid'].apply(lambda e: docid_encode_map.get(str(e),-1))
    source_data_tfidf['wordid3_encoded'] = source_data_tfidf['wordid3'].apply(lambda e: wordid_encode_map.get(str(e),-1))
    
    # 索引与词对应关系
    with open('./data/wordid_decode_map.csv', mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Encoded ID', 'Original Word'])  # 写入标题行 
        for encoded_id, original_word in wordid_decode_map.items():  # 遍历解码映射并写入每个条目
            writer.writerow([encoded_id, original_word])

    # 对于docid_decode_map也是类似的过程
    with open('./data/docid_decode_map.csv', mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Encoded ID', 'Original Doc ID'])
        for encoded_id, original_doc_id in docid_decode_map.items():
            writer.writerow([encoded_id, original_doc_id])
    
    # 统计唯一值的个数 
    wordid1_count = len(set(source_data_comatrix['wordid1_encoded'].values))
    print(wordid1_count)
    wordid2_count = len(set(source_data_comatrix['wordid2_encoded'].values))
    print(wordid2_count)
    docid_count = len(set(source_data_tfidf['docid_encoded'].values))
    print(docid_count)
    wordid3_count = len(set(source_data_tfidf['wordid3_encoded'].values))
    print(wordid3_count)
    
    # 保存编码后的矩阵
    final_source_data_comatrix = source_data_comatrix[['wordid1_encoded','wordid2_encoded','weight']].sort_values(by='wordid1_encoded', ascending=True)
    final_source_data_comatrix.to_csv(r'/home/lym/lab/project_work/project_versions/GCN_Link_Prediction_v3/data/final_source_data_comatrix_train.csv',index=False)
    final_source_data_tfidf = source_data_tfidf[['docid_encoded','wordid3_encoded','weight']].sort_values(by='docid_encoded', ascending=True)
    final_source_data_tfidf.to_csv(r'/home/lym/lab/project_work/project_versions/GCN_Link_Prediction_v3/data/final_source_data_tfidf_train.csv',index=False)
    
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

    graph_data = {
        ('word', 'co-occurrence', 'word'): (word_e_word_src, word_e_word_dst),
        ('word', 'co-occurrence_i', 'word'): (word_e_word_dst, word_e_word_src),
        ('doc', 'tf-idf', 'word'): (doc_e_word_src, doc_e_word_dst),
        ('word', 'tf-idf_i', 'doc'): (doc_e_word_dst, doc_e_word_src)
    }
    
    g = dgl.heterograph(graph_data)
    
    # 设置边特征
    g.edges['co-occurrence'].data['weight'] = torch.tensor(co_occurrence_weights, dtype=torch.float32)
    g.edges['tf-idf'].data['weight'] = torch.tensor(tfidf_weights, dtype=torch.float32)
    
    return g,word_e_word_count,doc_e_word_count

# ? 异构图上RGCN模型所需的自定义函数：RelGraphConvLayer、 RelGraphEmbed 以及 EntityClassify
# * 创建一个关系图卷积层
# * 参考文章：https://zhuanlan.zhihu.com/p/603389554
class RelGraphConvLayer(nn.Module):
    # 构造函数
    def __init__(self,
                in_feat,
                out_feat,
                rel_names,
                num_bases,  # 用于基础分解，减少参数数量
                 *,   # 用于指示位置参数的结束和仅关键字参数的开始
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

        # 这个地方只是起到计算的作用, 不保存数据
        self.conv = HeteroGraphConv({
            # graph conv 里面有模型参数weight,如果外边不传进去的话,里面新建
            # 相当于模型加了一层全链接, 对每一种类型的边计算卷积
            rel: GraphConv(in_feat, out_feat, norm='right', weight=False, bias=False)
            for rel in rel_names
        })

        self.use_weight = weight
        self.use_basis = num_bases < len(self.rel_names) and weight
        if self.use_weight:
            if self.use_basis:
                self.basis = dglnn.WeightBasis((in_feat, out_feat), num_bases, len(self.rel_names))  # 共享基础权重
            else:
                # 每个关系,有一个weight,全连接层，self.weight是一个三维的权重矩阵
                self.weight = nn.Parameter(torch.Tensor(len(self.rel_names), in_feat, out_feat))
                nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))  # 初始化一个神经网络层的权重参数

        # bias
        if bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_feat))  # 输出特征有偏置值
            nn.init.zeros_(self.h_bias)  # 训练开始时，偏置为0，不会对网络产生任何影响

        # weight for self loop
        if self.self_loop:  # 考虑自身的特征，为自身赋予权重
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    # 前向传播函数，处理图结构数据，并通过卷积操作生成节点的嵌入表示
    def forward(self, g, inputs):
        
        g = g.local_var()  # 创建图g的局部副本
        if self.use_weight:
            weight = self.basis() if self.use_basis else self.weight
            # 这每个关系对应一个权重矩阵对应输入维度和输出维度
            wdict = {self.rel_names[i]: {'weight': w.squeeze(0)}  # 去除第0维，如果它的大小为1，(1, A, B)变为(A, B)
                    for i, w in enumerate(torch.split(weight, 1, dim=0))}  # 分割多个子张量
        else:
            wdict = {}

        if g.is_block: # 处理图g是否为一个块结构，这通常在小批量训练中使用
            inputs_src = inputs
            inputs_dst = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}
        else:
            inputs_src = inputs_dst = inputs

        # 多类型的边结点卷积完成后的输出
        # 输入的是blocks 和 embeding
        hs = self.conv(g, inputs, mod_kwargs=wdict)
        
        # 在GCN上应用于每个节点类型的特征变换
        def _apply(ntype, h):
            if self.self_loop:
                h = h + torch.matmul(inputs_dst[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)

        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}

# * 用于无特征异构图的嵌入层，不依赖节点的初始特征
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

        # create weight embeddings for each node for each relation
        self.embeds = nn.ParameterDict()  # 字典，用于存储模型参数
        for ntype in g.ntypes:
            embed = nn.Parameter(torch.Tensor(g.number_of_nodes(ntype), self.embed_size))
            nn.init.xavier_uniform_(embed, gain=nn.init.calculate_gain('relu'))
            self.embeds[ntype] = embed
    # 前向传播函数，返回嵌入向量
    def forward(self, block=None):
        
        return self.embeds

# *用于对图中的实体进行分类
class EntityClassify(nn.Module):
    # 构造函数
    def __init__(self,
                g,
                h_dim, out_dim,  # 输出层的维度，通常对应于分类任务中的类别数
                num_bases=-1,  # 基数，用于控制关系类型的权重共享；如果小于0或大于关系类型的数量，则使用关系类型的数量作为基数
                num_hidden_layers=1,  # 双层
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
        # i2h
        self.layers.append(RelGraphConvLayer(
            self.h_dim, self.h_dim, self.rel_names,
            self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
            dropout=self.dropout, weight=False))  # 不使用可学习权重矩阵，可以减少参数，防止过拟合

        # h2h , 这里不添加隐层,只用2层卷积
        # for i in range(self.num_hidden_layers):
        #    self.layers.append(RelGraphConvLayer(
        #        self.h_dim, self.h_dim, self.rel_names,
        #        self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
        #        dropout=self.dropout))
        # h2o

        self.layers.append(RelGraphConvLayer(
            self.h_dim, self.out_dim, self.rel_names,
            self.num_bases, activation=None,
            self_loop=self.use_self_loop))

    # 输入 blocks,embeding
    def forward(self, h=None, blocks=None):
        if h is None:  # 无节点特征，self.embed_layer() 来获取图中所有节点的嵌入表示
            # full graph training
            h = self.embed_layer()
        if blocks is None:  
            # full graph training
            for layer in self.layers:
                h = layer(self.g, h)
        else:
            # minibatch training
            # 输入 blocks,embeding
            for layer, block in zip(self.layers, blocks):  # 创建一个迭代器，该迭代器会将两个或更多的可迭代对象中的元素按顺序配对成元组
                h = layer(block, h)
        return h

    # 用于在图神经网络中进行推理的，它处理整个图或其大型子图来生成节点的嵌入表示
    def inference(self, g, batch_size, device, num_workers=0, x=None):

        if x is None:
            x = self.embed_layer()

        for l, layer in enumerate(self.layers):  # 遍历每一层，张量的大小由图中该类型节点的数量和相应的特征维度决定
            y = {
                k: torch.zeros(
                    g.number_of_nodes(k),
                    self.h_dim if l != len(self.layers) - 1 else self.out_dim)
                for k in g.ntypes}

            
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.DataLoader(   # 将图 g 分成多个批次，每个批次包含一定数量的节点，目的：高效加载和处理图数据
                g,
                {k: torch.arange(g.number_of_nodes(k)) for k in g.ntypes},
                sampler,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=num_workers)
            
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):  # 循环处理 dataloader 生成的每个批次，对于每个批次，它将图块 block 和相应的节点特征 h 传递给当前层 layer 进行前向传播。计算后的特征 h 被收集到字典 y 中
                # print(input_nodes)
                block = blocks[0].to(device)
                        
                h = {k: x[k][input_nodes[k]].to(device) for k in input_nodes.keys()}
                h = layer(block, h)

                for k in h.keys():
                    y[k][output_nodes[k]] = h[k].cpu()  # h[k].cpu()

            x = y  # 更新 x 为最新的特征表示，以便在下一层中使用
        return y  # 图 g 中所有节点的最终特征表示
    
# ? 模型采样超参与边采样
# *根据节点类型和节点ID抽取embeding 参与模型训练更新，它从节点嵌入字典中提取特定节点的嵌入
def extract_embed(node_embed, input_nodes,device):
    emb = {}
    for ntype, nid in input_nodes.items():
        # 确保nid在正确的设备上
        nid = nid.to(device)
        nid = input_nodes[ntype]
        emb[ntype] = node_embed[ntype][nid]
    return emb

# ? 模型结构定义与损失函数说明
# * 用于预测异构图中边的得分
class HeteroDotProductPredictor(nn.Module):

    def forward(self, graph, h, etype):
        # 在计算之外更新h,保存为全局可用
        # h contains the node representations for each edge type computed from node_clf_hetero.py
        with graph.local_scope(): #  创建了一个本地作用域,在这个作用域内对图的任何修改都不会影响图的全局状态
            graph.ndata['h'] = h  # assigns 'h' of all node types in one shot
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']
        
# * 定义一个图神经网络模型，用于实体分类和边预测
class Model(nn.Module):

    def __init__(self, graph, hidden_feat_dim, out_feat_dim):
        super().__init__()
        self.rgcn = EntityClassify(graph,
                                hidden_feat_dim,
                                out_feat_dim)
        self.pred = HeteroDotProductPredictor()

    def forward(self, h, pos_g, neg_g, blocks, etype):
        h = self.rgcn(h, blocks)  # 更新节点特征h
        return self.pred(pos_g, h, etype), self.pred(neg_g, h, etype)  # 计算正样本图pos_g和负样本图neg_g中边的得分
    
    def inference(self, hetero_graph, batch_size, device, num_workers, x):
        # 调用 EntityClassify 实例的 inference 方法
        return self.rgcn.inference(hetero_graph, batch_size, device, num_workers, x)
    
# * 自定义的损失函数，用于图卷积网络（GCN）中的链接预测任务，最大化正样本边的得分和负样本边的得分之间的差距
class MarginLoss(nn.Module):

    def forward(self, pos_score, neg_score):
        # 求损失的平均值 , view 改变tensor 的形状
        # 1- pos_score + neg_score ,应该是 -pos 符号越大变成越小  +neg_score 越小越好
        return (1 - pos_score + neg_score.view(pos_score.shape[0], -1)).clamp(min=0).mean()    

# ? 定义一个函数来找出test词表中独有的词
def find_new_words(train_path, test_path):
    # 读取train词表文件，并将词存储在集合中以便快速查找
    with open(train_path, 'r') as file:
        train_vocabulary = set(file.read().splitlines())
    
    # 初始化一个空列表来存储独有的词
    unique_words = []
    
    # 读取test词表文件
    with open(test_path, 'r') as file:
        # 对于test词表中的每一个词
        for word in file:
            word = word.strip()  # 移除任何开始和结尾的空白字符
            # 如果这个词不在train词表中，则将其添加到独有词列表中
            if word not in train_vocabulary:
                unique_words.append(word)
    
    return unique_words
    
    
# ? 评估模型，用于测试
def evaluate_model(model, test_graph, test_labels):
    model.eval()
    with torch.no_grad():
        # 使用模型进行预测
        predictions = model(test_graph, test_graph.ndata['feat'])
        
        # 计算评估指标
        # 这里我们以 'co-occurrence' 边的 ROC AUC 分数为例
        pred_scores = predictions['co-occurrence'].sigmoid()
        roc_auc = roc_auc_score(test_labels['co-occurrence'].numpy(), pred_scores.numpy())
        
        return roc_auc
    
# ! 4_Comparative Experiment.py
# ? 为训练集和测试集添加图数据
def data_create(bidirectional_edges,device,word_tensor):
    edge_list_1 = []
    edge_list_2 = []

    for index, row in bidirectional_edges.iterrows():
        edge_list_1.append(int(row.iloc[0]))
        edge_list_2.append(int(row.iloc[1]))

    print(edge_list_1[:5])
    print(edge_list_2[:5])

    pos_num = len(edge_list_1)
    print('正样本数：',pos_num)

    arr = np.zeros((2,pos_num), dtype=np.int64)
    arr[0] = edge_list_1
    arr[1] = edge_list_2
    edge_index = torch.from_numpy(arr)
    edge_index = edge_index.to(device)

    data_wos = Data(x = word_tensor, edge_index = edge_index)    
    
    return data_wos

# ? 求指标的函数
def get_scores(clf, X_new, y_new): # 接受一个分类器和测试数据，然后返回多个评分指标
    scoring = ['precision', 'recall', 'accuracy', 'roc_auc', 'f1', 'average_precision']
    scorers = {scorer_name: check_scoring(clf, scorer_name) for scorer_name in scoring}
    #print(scorers)
    scores = multimetric_score(clf, X_new, y_new, scorers)
    return scores

def multimetric_score(estimator, X_test, y_test, scorers): # 函数计算并返回多个评分指标的字典
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