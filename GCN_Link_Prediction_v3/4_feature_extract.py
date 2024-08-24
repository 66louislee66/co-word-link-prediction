# LDA特征的提取
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import lda
import lda.datasets

# *对词表所有语料进行向量化
decode_map_df = pd.read_csv(r'./data/3_wordid_decode_map.csv')
feature_names = decode_map_df['Original Word'].tolist()
df = pd.read_csv(r'./data/1_webofsci_train_allclean.txt', header=None, sep = '\0')
documents = df[0].values.tolist()
count_vectorizer = CountVectorizer(vocabulary = feature_names)
cv = count_vectorizer.fit_transform(documents)
matrix = cv.toarray()

# * 训练LDA模型，主要为了得到topic_word权重矩阵，作为keyword的特征，得到LDA特征
# ! 参考来源：https://blog.csdn.net/m0_37052320/article/details/79117448
titles = documents
vocab = feature_names
x = matrix

model = lda.LDA(n_topics = 20, n_iter = 1500, random_state = 1)  #设置主题topic的数量是20
model.fit(x)

topic_word = model.topic_word_  #下面这个topic_word使我们主要想要的keywords的特征

n_top_words = 8  #下面主要实现一些结果，n_top_words是设定现实的每个主题的高频词的个数
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))
doc_topic = model.doc_topic_
for i in range(10):
    print("{} (top topic: {})".format(titles[i], doc_topic[i].argmax()))

columns = feature_names
df_output = pd.DataFrame(topic_word, columns = columns)
print(df_output.shape)

vectorized_func = np.vectorize(lambda x: 1 if x > 0.001 else 0)
df_output = df_output.apply(vectorized_func)
print(df_output.shape)

# * 添加TF-IDF特征
# ! 参考来源：https://zhuanlan.zhihu.com/p/448623822
tf_idf_vectorizer = TfidfVectorizer(vocabulary = feature_names)
tf_idf = tf_idf_vectorizer.fit_transform(df[0])
matrix_keywords_words = tf_idf.toarray()
pd_data = pd.DataFrame(matrix_keywords_words, columns = columns)
my_final_df = pd_data.append(df_output, ignore_index = True)
my_final_df.to_csv(r'./data/4_webofsci_train_LDA_features.csv')
print(df_output.shape)