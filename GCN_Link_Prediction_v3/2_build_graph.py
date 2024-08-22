# 构建TF-IDF矩阵和共现矩阵
# * 导入模块
import scipy.sparse as sp  
from utils import find_new_words
from sklearn.feature_extraction.text import TfidfVectorizer  
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from collections import Counter
import re
import yake

dataset = ["train", "test"]
for data_type in dataset:
    
    # * 初始化YAKE模型的参数
    language = "en" # 文档语言
    max_ngram_size = 1 # N-grams
    deduplication_thresold = 0.9 # 筛选阈值,越小关键词越少
    deduplication_algo = 'seqm'
    windowSize = 1
    numOfKeywords = 20 # 最大数量
    all_keywords = []  # 存储所有关键词及其概率值
    num_keywords = 500  # 关键词的数量

    # * 读取每行文本内容
    file_path = r'/home/lym/lab/project_work/project_versions/GCN_Link_Prediction_v3/data/1_webofsci_{}_clean.txt'.format(data_type)
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # * 文本转换为字符串形式，并利用YAKE提取关键词
    for line in lines:
        text = line.strip()
        kw_extractor = yake.KeywordExtractor(lan=language, 
                                         n=max_ngram_size, 
                                         dedupLim=deduplication_thresold, 
                                         dedupFunc=deduplication_algo, 
                                         windowsSize=windowSize, 
                                         top=numOfKeywords)
        keywords = kw_extractor.extract_keywords(text)
        all_keywords.extend(keywords)

    # * 去除重复的词，仅保留概率最大的词，同时提取排名前300的关键词并保存
    # ? 词表的词数能否作为超参数
    keyword_dict = {}
    for keyword, score in all_keywords:
        keyword = keyword.lower()
        if re.search(r"[()\:\;,.']", keyword):
            continue
        if keyword in keyword_dict:
            if score > keyword_dict[keyword]:  # 保留概率值较大的关键词
                keyword_dict[keyword] = score
        else:
            keyword_dict[keyword] = score

    final_keywords = sorted(keyword_dict.items(), key=lambda x: x[1], reverse=True)  # 将字典转换回列表并排序

    top_keywords = final_keywords[:num_keywords]  # 根据概率值排序并取前300个关键词

    output_file_path = r'/home/lym/lab/project_work/project_versions/GCN_Link_Prediction_v3/data/2_top_keywords_{}.txt'.format(data_type)
    with open(output_file_path, 'w') as output_file:
        for keyword, score in top_keywords:
            output_file.write(f"{keyword}\t{score}\n")

    # * 构建词表vocab并保存，即排名前300的关键词
    vocab = [kw[0].lower() for kw in top_keywords]
    vocab_size = len(vocab)

    vocab_str = '\n'.join(vocab)
    with open(r'/home/lym/lab/project_work/project_versions/GCN_Link_Prediction_v3/data/2_webofsci_vocabulary_{}.txt'.format(data_type), "w", encoding='UTF-8') as output_file:
        output_file.write(vocab_str)

    # * 构建文档和词之间的TF-IDF关系matrix_keywords_words：衡量词的重要程度
    # ! 注意：词表中的词必须能在你的文档中找到，不然会出现错误
    df = pd.read_csv(r'/home/lym/lab/project_work/project_versions/GCN_Link_Prediction_v3/data/1_webofsci_{}_allclean.txt'.format(data_type), header=None, sep = '\0')
    tf_idf_vectorizer = TfidfVectorizer(vocabulary = vocab)
    tf_idf = tf_idf_vectorizer.fit_transform(df[0])
    matrix_keywords_words = tf_idf.toarray()
    columns = vocab
    pd_data = pd.DataFrame(matrix_keywords_words, columns = columns)
    pd_data.to_csv(r'/home/lym/lab/project_work/project_versions/GCN_Link_Prediction_v3/data/2_webofsci_tf-idf_features_{}.csv'.format(data_type))

    # * 构建词与词之间的共现关系df_comatrix：统计每个词汇与其他词汇在同一个文档中共同出现的次数
    doc_list = []
    for doc_words in df[0]:
        doc_list.append(doc_words.split())
    
    comatrix_sparse = lil_matrix((len(vocab), len(vocab)), dtype=np.int32)  # 创建一个LIL格式的稀疏矩阵
    
    word_to_index = {word: i for i, word in enumerate(vocab)}  # 构建词汇索引映射
    
    # ! 注意：由于文档中的词的集合要大于我的词表，因此，当你对比一篇文档中的两个词，你还需要确定是否在词表中
    for doc_words in doc_list:  # 填充共现矩阵
        word_counts = Counter(doc_words)
        for word1, count1 in word_counts.items():
            for word2, count2 in word_counts.items():
                if word1 != word2 and word1 in word_to_index and word2 in word_to_index:
                    comatrix_sparse[word_to_index[word1], word_to_index[word2]] += min(count1, count2)
    
    comatrix_csr = comatrix_sparse.tocsr()
    comatrix_dense = comatrix_csr.toarray()
    df_comatrix = pd.DataFrame(comatrix_dense, index=vocab, columns=vocab)
    
    # * 处理测试集中的共现矩阵
    if data_type == "test":
        vectorized_func = np.vectorize(lambda x: 1 if x > 0 else 0)
        df_comatrix = df_comatrix.apply(vectorized_func)

        train_path = '/home/lym/lab/project_work/project_versions/GCN_Link_Prediction_v3/data/2_webofsci_vocabulary_train.txt'
        test_path = '/home/lym/lab/project_work/project_versions/GCN_Link_Prediction_v3/data/2_webofsci_vocabulary_test.txt'

        new_words = find_new_words(train_path, test_path)

        print(f"Test词表中独有的词: {new_words}")
        print(len(new_words))
        
        df_comatrix = df_comatrix.drop(columns=new_words, errors='ignore')
        df_comatrix = df_comatrix.drop(index=new_words, errors='ignore')
        
        df_comatrix.to_csv(r'/home/lym/lab/project_work/project_versions/GCN_Link_Prediction_v3/data/2_comatrix_{}_label.csv'.format(data_type))

        word_num_test = df_comatrix.shape[0]
        doc_num_test = pd_data.shape[0]
        
    if data_type == "train":
        df_comatrix.to_csv(r'/home/lym/lab/project_work/project_versions/GCN_Link_Prediction_v3/data/2_comatrix_{}.csv'.format(data_type))
        
        # * 对词共现关系矩阵进行双向归一化并保存
        # ! 注意：如果某些点无任何连接，那么它的度就为0，导致度矩阵存在零值，影响计算，可以将零值转为一个非常小的值
        comatrix = sp.csr_matrix(df_comatrix.values)  
        degrees = np.array(comatrix.sum(1)).flatten()  
        np.savetxt(r'/home/lym/lab/project_work/project_versions/GCN_Link_Prediction_v3/data/2_degrees_{}.csv'.format(data_type), degrees, delimiter=',')
        degrees[degrees == 0] = 1e-10  # 将度数为0的值转为一个非常小的值
        D_inv_sqrt = sp.diags(np.power(degrees,-0.5)) # 计算度矩阵D的逆平方根
        comatrix_normalized = D_inv_sqrt @ comatrix @ D_inv_sqrt  # 计算双向归一化的邻接矩阵
        df_comatrix_normalized = pd.DataFrame(comatrix_normalized.toarray(), index=vocab, columns=vocab)
        df_comatrix_normalized.to_csv(r'/home/lym/lab/project_work/project_versions/GCN_Link_Prediction_v3/data/2_comatrix_normalized_{}.csv'.format(data_type))

        # * 数据准备（提取非零元素的节点对应行列名称）并保存
        df_comatrix_normalized = pd.read_csv(r'/home/lym/lab/project_work/project_versions/GCN_Link_Prediction_v3/data/2_comatrix_normalized_{}.csv'.format(data_type), index_col=0)
        pd_data = pd.read_csv(r'/home/lym/lab/project_work/project_versions/GCN_Link_Prediction_v3/data/2_webofsci_tf-idf_features_{}.csv'.format(data_type), index_col=0)

        source_data_comatrix = pd.DataFrame({
            'wordid1': df_comatrix_normalized[df_comatrix_normalized != 0].stack().index.get_level_values(0),
            'wordid2': df_comatrix_normalized[df_comatrix_normalized != 0].stack().index.get_level_values(1),
            'weight': df_comatrix_normalized[df_comatrix_normalized != 0].stack().values
        })

        source_data_tfidf = pd.DataFrame({
            'docid': pd_data[pd_data != 0].stack().index.get_level_values(0),
            'wordid3': pd_data[pd_data != 0].stack().index.get_level_values(1),
            'weight': pd_data[pd_data != 0].stack().values
        })
        
        source_data_comatrix.to_csv(r'/home/lym/lab/project_work/project_versions/GCN_Link_Prediction_v3/data/2_source_data_comatrix_{}.csv'.format(data_type), index=False)
        source_data_tfidf.to_csv(r'/home/lym/lab/project_work/project_versions/GCN_Link_Prediction_v3/data/2_source_data_tfidf_{}.csv'.format(data_type), index=False)
        
        # word_num_train = df_comatrix.shape[0]
        # doc_num_train = pd_data.shape[0]

# print('训练集：')
# print('词数：', word_num_train)
# print('文档数：', doc_num_train)
# print('测试集：')
# print('词数：', word_num_test)
# print('文档数：', doc_num_test)