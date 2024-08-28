# 对比传统机器学习算法
# * 导入模块
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges
import torch.optim as optim
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
import torch.nn.functional as F
from sklearn import svm, linear_model, neighbors
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import tree, ensemble
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve
from utils import *
import gc
balanced = {0: 8, 1: 1}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

# * 定义模型实例
nb_model = GaussianNB()
lr_model = linear_model.LogisticRegression(solver='lbfgs',max_iter=1000,class_weight=balanced)
rf_model = ensemble.RandomForestClassifier(n_estimators=50, n_jobs=1,class_weight=balanced)
xgb_model = XGBClassifier(learning_rate=0.1, n_estimators=50, max_depth=6, subsample=0.8, colsample_bytree=0.8,random_state=27, scale_pos_weight=0.4)
svm_model = LinearSVC(C=1.0, random_state=42, dual=False,class_weight=balanced)

# clfs = [('Naive Bayes',nb_model),('Logistic Regression',lr_model),('Random Forest',rf_model), ('XGB',xgb_model), ('SVM', svm_model)]
clfs = [('Naive Bayes',nb_model)]

# * 训练和测试模型的主函数
def main(clfs):
    # Training dataset
    neg_edge_index = negative_sampling(
        edge_index=data_train.edge_index,
        num_nodes=data_train.num_nodes,
        num_neg_samples=data_train.edge_index.size(1))
    
    num_pos_samples = data_train.edge_index.shape[1]
    num_neg_samples = neg_edge_index.shape[1]
    # 初始化训练集X，y矩阵并构建
    x_arr = np.zeros((num_pos_samples + num_neg_samples, data_train.x.shape[1] * 2), dtype=np.float32)
    y_arr = np.zeros((num_pos_samples + num_neg_samples), dtype=np.float32)
    
    print(y_arr.shape)
    print(y_arr)
    print("*****")
    
    for i in range(num_pos_samples):
        x_arr[i] = np.array(LDA_tensor[data_train.edge_index[0][i]].detach().cpu().numpy().tolist() + LDA_tensor[data_train.edge_index[1][i]].detach().cpu().numpy().tolist())
        y_arr[i] = 1.
    for i in range(num_neg_samples):
        x_arr[i+num_pos_samples] = np.array(LDA_tensor[neg_edge_index[0][i]].detach().cpu().numpy().tolist() + LDA_tensor[neg_edge_index[1][i]].detach().cpu().numpy().tolist())
    
    print(y_arr.shape)
    print(y_arr)
    print("#####")
        
    # Testing dataset
    neg_edge_index_test = negative_sampling(
        edge_index=data_test.edge_index,
        num_nodes=data_test.num_nodes,                                                                                                               
        num_neg_samples=data_test.edge_index.size(1))
    
    num_pos_samples_test = data_test.edge_index.shape[1]
    num_neg_samples_test = neg_edge_index_test.shape[1]
    
    # 初始化测试集X, y矩阵并构建
    x_arr_test = np.zeros((num_pos_samples_test + num_neg_samples_test, data_test.x.shape[1] * 2), dtype=np.float32)
    y_arr_test = np.zeros((num_pos_samples_test + num_neg_samples_test), dtype=np.float32)
    for i in range(num_pos_samples_test):
        x_arr_test[i] = np.array(LDA_tensor[data_test.edge_index[0][i]].detach().cpu().numpy().tolist() + LDA_tensor[data_test.edge_index[1][i]].detach().cpu().numpy().tolist())
        y_arr_test[i] = 1.
    for i in range(num_neg_samples_test):
        x_arr_test[i+data_test.edge_index.shape[1]] = np.array(LDA_tensor[neg_edge_index_test[0][i]].detach().cpu().numpy().tolist() + LDA_tensor[neg_edge_index_test[1][i]].detach().cpu().numpy().tolist())

    print(y_arr.shape)
    print(y_arr)
    print("&&&&&")
    
    results = pd.DataFrame()
    for name, clf in clfs:
        clf.fit(x_arr, y_arr)
        print("fit successed!")
        # 训练集上的评估
        train_scores = get_scores(clf, x_arr, y_arr)
        print("Training scores is: ", train_scores)
        train_scores["method"] = name + "_train"
        results = pd.concat([results, pd.DataFrame([train_scores])], ignore_index=True)
        
        # 测试集上的评估
        test_scores = get_scores(clf, x_arr_test, y_arr_test)
        print("Test score is: ", test_scores)
        test_scores['method'] = name + "_test"
        results = pd.concat([results, pd.DataFrame([test_scores])], ignore_index=True)
        del clf
        gc.collect()
    return results

results = main(clfs)
results.to_csv(r'./data/4_results_Comparative_Experiment.csv', index=False)
