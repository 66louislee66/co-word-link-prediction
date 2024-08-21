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
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import tree, ensemble
from xgboost import XGBClassifier
from utils import *
import gc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ? 加载节点数据(通过自编码器训练出低维的特征向量)
# df_word = pd.read_csv(r'/home/lym/lab/project_work/project_versions/GCN_Link_Prediction_v3/data/word_embeddings_inference.csv')
# scaler = StandardScaler()
# df_word_standardized = pd.DataFrame(scaler.fit_transform(df_word), columns=df_word.columns)
# df_word_standardized.to_csv(r'/home/lym/lab/project_work/project_versions/GCN_Link_Prediction_v3/data/word_embeddings_inference_standardized.csv', index=False)
# df_word_standardized = df_word_standardized.astype(np.float32)
# word_arr = df_word_standardized.values
# word_tensor = torch.from_numpy(word_arr)
# word_tensor = word_tensor.to(device)
compressed_features = pd.read_csv(r'/home/lym/lab/project_work/project_versions/GCN_Link_Prediction_v3/data/compressed_features.csv', index_col = 0)
compressed_features = compressed_features.astype(np.float32)
LDA_arr = compressed_features.values
LDA_tensor = torch.from_numpy(LDA_arr)
LDA_tensor = LDA_tensor.to(device)

# ? 训练集添加图数据
final_source_data_comatrix = pd.read_csv(r'/home/lym/lab/project_work/project_versions/GCN_Link_Prediction_v3/data/final_source_data_comatrix_train.csv')
reverse_edges = final_source_data_comatrix[['wordid2_encoded', 'wordid1_encoded', 'weight']].copy()
reverse_edges.columns = ['wordid1_encoded', 'wordid2_encoded', 'weight']
bidirectional_edges = pd.concat([final_source_data_comatrix, reverse_edges], ignore_index=True)
bidirectional_edges = bidirectional_edges.sort_values(by='wordid1_encoded', ascending=True)

data_train = data_create(bidirectional_edges,device,LDA_tensor)

# ? 测试集添加图数据
comatix_test_label = pd.read_csv(r'/home/lym/lab/project_work/project_versions/GCN_Link_Prediction_v3/data/comatrix_test_label.csv', index_col=0)

# 将DataFrame转换为长格式，并保留原始的行和列标签
stacked_comatrix = comatix_test_label.stack().reset_index()

# 重命名列以更清晰地表示数据
stacked_comatrix.columns = ['row_word', 'col_word', 'label']

# 过滤出上三角部分的数据（不包括对角线）
upper_tri = stacked_comatrix[stacked_comatrix['row_word'] < stacked_comatrix['col_word']]

# 分离正样本
balanced_data = upper_tri[upper_tri['label'] == 1]

# 加载 wordid_decode_map.csv 文件
wordid_decode_df = pd.read_csv('./data/wordid_decode_map.csv')

# 创建从词名称到索引的映射字典
word_to_index = dict(zip(wordid_decode_df['Original Word'], wordid_decode_df['Encoded ID']))

# 假设 balanced_data 是您已经有的DataFrame，其中包含词的名称
# 替换 'row_word' 和 'col_word' 列中的词名称为索引
balanced_data.loc[:, 'row_word'] = balanced_data['row_word'].map(word_to_index)
balanced_data.loc[:, 'col_word'] = balanced_data['col_word'].map(word_to_index) 

reverse_edges_test = balanced_data[['col_word', 'row_word', 'label']].copy()
reverse_edges_test.columns = ['row_word', 'col_word', 'label']
bidirectional_edges_test = pd.concat([balanced_data, reverse_edges_test], ignore_index=True)
bidirectional_edges_test = bidirectional_edges_test.sort_values(by='row_word', ascending=True)

data_test = data_create(bidirectional_edges = bidirectional_edges_test, device = device,word_tensor = LDA_tensor)

# ? 定义模型实例
nb_model = GaussianNB()
lr_model = linear_model.LogisticRegression(solver='lbfgs')
rf_model = ensemble.RandomForestClassifier(n_estimators=50, n_jobs=1)
xgb_model = XGBClassifier(learning_rate=0.1, n_estimators=50, max_depth=6, subsample=0.8, colsample_bytree=0.8,random_state=27)
svm_model = SVC(kernel='linear', C=1.0, random_state=42)

# clfs = [('Naive Bayes',nb_model),('Logistic Regression',lr_model),('Random Forest',rf_model), ('XGB',xgb_model)]
clfs = [('SVM',svm_model)]

# ? 训练和测试模型的主函数
def main(clfs):
    # Training dataset
    neg_edge_index = negative_sampling(
        edge_index=data_train.edge_index,
        num_nodes=data_train.num_nodes,
        num_neg_samples=data_train.edge_index.size(1))
    
    # 初始化训练集X，y矩阵并构建
    x_arr = np.zeros((data_train.edge_index.shape[1] * 2, data_train.x.shape[1] * 2), dtype=np.float32)
    y_arr = np.zeros((data_train.edge_index.shape[1] * 2), dtype=np.float32)
    
    print(y_arr.shape)
    print(y_arr)
    print("*****")
    
    for i in range(data_train.edge_index.shape[1]):
        x_arr[i] = np.array(LDA_tensor[data_train.edge_index[0][i]].detach().cpu().numpy().tolist() + LDA_tensor[data_train.edge_index[1][i]].detach().cpu().numpy().tolist())
    for i in range(neg_edge_index.shape[1]):
        x_arr[i+data_train.edge_index.shape[1]] = np.array(LDA_tensor[neg_edge_index[0][i]].detach().cpu().numpy().tolist() + LDA_tensor[neg_edge_index[1][i]].detach().cpu().numpy().tolist())
    for i in range(data_train.edge_index.shape[1]):
        y_arr[i] = 1.
    print(y_arr.shape)
    print(y_arr)
    print("#####")
        
    # Testing dataset
    neg_edge_index_test = negative_sampling(
        edge_index=data_test.edge_index,
        num_nodes=data_test.num_nodes,                                                                                                               
        num_neg_samples=data_test.edge_index.size(1))
    
    # 初始化测试集X, y矩阵并构建
    x_arr_test = np.zeros((data_test.edge_index.shape[1] * 2, data_test.x.shape[1] * 2), dtype=np.float32)
    y_arr_test = np.zeros((data_test.edge_index.shape[1] * 2), dtype=np.float32)
    for i in range(data_test.edge_index.shape[1]):
        x_arr_test[i] = np.array(LDA_tensor[data_test.edge_index[0][i]].detach().cpu().numpy().tolist() + LDA_tensor[data_test.edge_index[1][i]].detach().cpu().numpy().tolist())
    for i in range(neg_edge_index_test.shape[1]):
        x_arr_test[i+data_test.edge_index.shape[1]] = np.array(LDA_tensor[neg_edge_index_test[0][i]].detach().cpu().numpy().tolist() + LDA_tensor[neg_edge_index_test[1][i]].detach().cpu().numpy().tolist())
    for i in range(data_test.edge_index.shape[1]):
        y_arr_test[i] = 1.
    print(y_arr.shape)
    print(y_arr)
    print("&&&&&")
    
    results = pd.DataFrame()
    for name, clf in clfs:
        clf.fit(x_arr, y_arr)
        print("fit successed!")
        scores = get_scores(clf, x_arr_test, y_arr_test)
        print("score is: ", scores)
        scores['method'] = name
        results = pd.concat([results, pd.DataFrame([scores])], ignore_index=True)
        del clf
        gc.collect()
    return results

results = main(clfs)
results.to_csv(r'/home/lym/lab/project_work/project_versions/GCN_Link_Prediction_v3/data/results_Comparative_Experiment.csv', index=False)