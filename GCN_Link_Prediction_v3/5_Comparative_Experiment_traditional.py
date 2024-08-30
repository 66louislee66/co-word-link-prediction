# 对比传统机器学习算法
# * 导入模块
import pandas as pd
import numpy as np
import torch
from torch_geometric.utils import negative_sampling
from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn import ensemble
from xgboost import XGBClassifier
from utils import *
import gc
# balanced = {0: 8, 1: 1}
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# * 加载节点数据(LDA特征)
dataset = ["train", "test"]
for data_type in dataset:
    df_LDA = pd.read_csv(r'./data/4_webofsci_LDA_features_{}.csv'.format(data_type), index_col = 0)
    df_LDA = df_LDA.astype(np.float32)
    LDA_arr = df_LDA.values.T
    LDA_tensor = torch.from_numpy(LDA_arr)
    if data_type == "train":
        LDA_tensor_train = LDA_tensor.to(device)
        print(LDA_tensor_train.shape)
    else:
        LDA_tensor_test = LDA_tensor.to(device)
        print(LDA_tensor_test.shape)

# * 训练集添加图数据
source_data_comatrix = pd.read_csv(r'./data/2_source_data_comatrix_train.csv')
reverse_edges = source_data_comatrix[['wordid2', 'wordid1', 'weight']].copy()
reverse_edges.columns = ['wordid1', 'wordid2', 'weight']
bidirectional_edges = pd.concat([source_data_comatrix, reverse_edges], ignore_index=True)

vocabulary_train = pd.read_csv(r'./data/2_webofsci_vocabulary_train.txt', header=None, names=['word'])
word_to_index_train = {word: idx for idx, word in enumerate(vocabulary_train['word'])}
bidirectional_edges['wordid1'] = bidirectional_edges['wordid1'].map(word_to_index_train)
bidirectional_edges['wordid2'] = bidirectional_edges['wordid2'].map(word_to_index_train)

bidirectional_edges = bidirectional_edges.sort_values(by='wordid1', ascending=True)

data_train = data_create(bidirectional_edges,device,LDA_tensor_train)

# * 测试集添加图数据
comatix_test_label = pd.read_csv(r'./data/2_comatrix_test_label.csv', index_col=0)
stacked_comatrix = comatix_test_label.stack().reset_index()
stacked_comatrix.columns = ['row_word', 'col_word', 'label']

upper_tri = stacked_comatrix[stacked_comatrix['row_word'] < stacked_comatrix['col_word']]

balanced_data = upper_tri[upper_tri['label'] == 1] 

vocabulary_test = pd.read_csv(r'./data/2_webofsci_vocabulary_test.txt', header=None, names=['word'])
word_to_index_test = {word: idx for idx, word in enumerate(vocabulary_test['word'])}

balanced_data.loc[:, 'row_word'] = balanced_data['row_word'].map(word_to_index_test)
balanced_data.loc[:, 'col_word'] = balanced_data['col_word'].map(word_to_index_test) 

reverse_edges_test = balanced_data[['col_word', 'row_word', 'label']].copy()
reverse_edges_test.columns = ['row_word', 'col_word', 'label']
bidirectional_edges_test = pd.concat([balanced_data, reverse_edges_test], ignore_index=True)
bidirectional_edges_test = bidirectional_edges_test.sort_values(by='row_word', ascending=True)

data_test = data_create(bidirectional_edges_test,device,LDA_tensor_test)

# * 定义模型实例
nb_model = GaussianNB()
lr_model = linear_model.LogisticRegression(solver='lbfgs',max_iter=1000)  # ,class_weight=balanced
rf_model = ensemble.RandomForestClassifier(n_estimators=50, n_jobs=1)
xgb_model = XGBClassifier(learning_rate=0.1, n_estimators=50, max_depth=6, subsample=0.8, colsample_bytree=0.8,random_state=27) # , scale_pos_weight=0.4
svm_model = LinearSVC(C=1.0, random_state=42, dual=False)

clfs = [('Naive Bayes',nb_model),('Logistic Regression',lr_model),('Random Forest',rf_model), ('XGB',xgb_model), ('SVM', svm_model)]
# clfs = [('XGB',xgb_model)]

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
        x_arr[i] = np.array(LDA_tensor_train[data_train.edge_index[0][i]].detach().cpu().numpy().tolist() + LDA_tensor_train[data_train.edge_index[1][i]].detach().cpu().numpy().tolist())
        y_arr[i] = 1.
    for i in range(num_neg_samples):
        x_arr[i+num_pos_samples] = np.array(LDA_tensor_train[neg_edge_index[0][i]].detach().cpu().numpy().tolist() + LDA_tensor_train[neg_edge_index[1][i]].detach().cpu().numpy().tolist())
    
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
        x_arr_test[i] = np.array(LDA_tensor_test[data_test.edge_index[0][i]].detach().cpu().numpy().tolist() + LDA_tensor_test[data_test.edge_index[1][i]].detach().cpu().numpy().tolist())
        y_arr_test[i] = 1.
    for i in range(num_neg_samples_test):
        x_arr_test[i+data_test.edge_index.shape[1]] = np.array(LDA_tensor_test[neg_edge_index_test[0][i]].detach().cpu().numpy().tolist() + LDA_tensor_test[neg_edge_index_test[1][i]].detach().cpu().numpy().tolist())

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
if len(clfs) == 1:
    results.to_csv(r'./data/5_Comparative_Experiment_{}.csv'.format(clfs[0][0]), index=False)
else:
    results.to_csv(r'./data/5_Comparative_Experiment_results.csv', index=False)
