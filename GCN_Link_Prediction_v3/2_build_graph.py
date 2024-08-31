# 构建TF-IDF矩阵、LDA矩阵和共现矩阵
# * 导入模块
import scipy.sparse as sp  
from utils import find_words
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
    language = "en" 
    max_ngram_size = 1 
    deduplication_thresold = 0.9
    deduplication_algo = 'seqm'
    windowSize = 1
    numOfKeywords = 20 
    all_keywords = []
    num_keywords = 500

    # * 读取每行文本内容
    file_path = r'./data/1_webofsci_{}_clean.txt'.format(data_type)
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

    # * 去除重复的词，仅保留概率最大的词，同时提取排名前num_keywords的关键词并保存
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

    final_keywords = sorted(keyword_dict.items(), key=lambda x: x[1], reverse=True)

    top_keywords = final_keywords[:num_keywords] 

    output_file_path = r'./data/2_top_keywords_{}.txt'.format(data_type)
    with open(output_file_path, 'w') as output_file:
        for keyword, score in top_keywords:
            output_file.write(f"{keyword}\t{score}\n")

    # * 构建词表vocab并保存
    vocab = [kw[0].lower() for kw in top_keywords]
    vocab_size = len(vocab)

    vocab_str = '\n'.join(vocab)
    with open(r'./data/2_webofsci_vocabulary_{}.txt'.format(data_type), "w", encoding='UTF-8') as output_file:
        output_file.write(vocab_str)

    # * 构建文档和词之间的TF-IDF关系matrix_keywords_words：衡量词的重要程度
    df = pd.read_csv(r'./data/1_webofsci_{}_allclean.txt'.format(data_type), header=None, sep = '\0')
    tf_idf_vectorizer = TfidfVectorizer(vocabulary = vocab, token_pattern=r'(?u)\b\w[\w-]*\b')
    tf_idf = tf_idf_vectorizer.fit_transform(df[0] )
    matrix_keywords_words = tf_idf.toarray()
    columns = vocab
    pd_data = pd.DataFrame(matrix_keywords_words, columns = columns)
    pd_data.to_csv(r'./data/2_webofsci_tf-idf_features_{}.csv'.format(data_type))

    # * 构建主题和词之间关系矩阵：统计词与主题的关联度
    pd_LDA = pd.read_csv(r'./data/2_webofsci_LDA_features_{}.csv'.format(data_type), index_col=0)

    # * 构建词与词之间的共现关系df_comatrix：统计每个词汇与其他词汇在同一个文档中共同出现的次数
    doc_list = []
    for doc_words in df[0]:
        doc_list.append(doc_words.split())
    
    comatrix_sparse = lil_matrix((len(vocab), len(vocab)), dtype=np.int32) 
    
    word_to_index = {word: i for i, word in enumerate(vocab)} 
    
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
        df_comatrix.to_csv(r'./data/2_comatrix_{}.csv'.format(data_type))

        train_path = './data/2_webofsci_vocabulary_train.txt'
        test_path = './data/2_webofsci_vocabulary_test.txt'

        words = find_words(train_path, test_path)
        df_comatrix = df_comatrix.drop(columns=words, errors='ignore')
        df_comatrix = df_comatrix.drop(index=words, errors='ignore')
        
        df_comatrix.to_csv(r'./data/2_comatrix_{}_label.csv'.format(data_type))

        vocabulary_test = df_comatrix.columns.tolist()
        with open(r'./data/2_webofsci_vocabulary_test_label.txt', 'w') as f:
            for word in vocabulary_test:
                f.write(f"{word}\n")

        word_num_test = df_comatrix.shape[0]
        doc_num_test = pd_data.shape[0]
        
    if data_type == "train":
        df_comatrix.to_csv(r'./data/2_comatrix_{}.csv'.format(data_type))
        # * 对词共现关系矩阵进行双向归一化并保存
        comatrix = sp.csr_matrix(df_comatrix.values)  
        degrees = np.array(comatrix.sum(1)).flatten()  
        np.savetxt(r'./data/2_degrees_{}.csv'.format(data_type), degrees, delimiter=',')
        degrees[degrees == 0] = 1e-10 
        D_inv_sqrt = sp.diags(np.power(degrees,-0.5))
        comatrix_normalized = D_inv_sqrt @ comatrix @ D_inv_sqrt  # 计算双向归一化的邻接矩阵
        df_comatrix_normalized = pd.DataFrame(comatrix_normalized.toarray(), index=vocab, columns=vocab)
        df_comatrix_normalized.to_csv(r'./data/2_comatrix_normalized_{}.csv'.format(data_type))

        # * 数据准备（提取非零元素的节点对应行列名称）并保存
        df_comatrix_normalized = pd.read_csv(r'./data/2_comatrix_normalized_{}.csv'.format(data_type), index_col=0)
        pd_data = pd.read_csv(r'./data/2_webofsci_tf-idf_features_{}.csv'.format(data_type), index_col=0)
        pd_LDA = pd.read_csv(r'./data/2_webofsci_LDA_features_{}.csv'.format(data_type), index_col=0) #1
        
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
        
        source_data_LDA = pd.DataFrame({
            'topicid': pd_LDA[pd_LDA != 0].stack().index.get_level_values(0),
            'wordid4': pd_LDA[pd_LDA != 0].stack().index.get_level_values(1),
            'weight': pd_LDA[pd_LDA != 0].stack().values
        })  #1
        
        source_data_comatrix.to_csv(r'./data/2_source_data_comatrix_{}.csv'.format(data_type), index=False)
        source_data_tfidf.to_csv(r'./data/2_source_data_tfidf_{}.csv'.format(data_type), index=False)
        source_data_LDA.to_csv(r'./data/2_source_data_LDA_{}.csv'.format(data_type), index=False)  #1
