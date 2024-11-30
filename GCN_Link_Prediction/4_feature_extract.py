# Extraction of LDA features
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import lda
import lda.datasets
import random

dataset = ["train", "test"]
for data_type in dataset:
    
    # * Vectorisation of all corpus of word lists
    vocabulary_df = pd.read_csv(r'./data/2_webofsci_vocabulary_{}.txt'.format(data_type), header=None, names=['Word'])
    feature_names = vocabulary_df['Word'].tolist()
    df = pd.read_csv(r'./data/1_webofsci_{}_allclean.txt'.format(data_type), header=None, sep = '\0')

    documents = df[0].values.tolist()
    count_vectorizer = CountVectorizer(vocabulary = feature_names)
    cv = count_vectorizer.fit_transform(documents)
    matrix = cv.toarray()

    # * Train the LDA model, mainly to get the topic_word weight matrix as a feature of keyword to get the LDA features
    titles = documents
    vocab = feature_names
    x = matrix

    model = lda.LDA(n_topics = 20, n_iter = 1500, random_state = 1)  # Set the number of topic topics to 20
    model.fit(x)

    topic_word = model.topic_word_  

    n_top_words = 8  
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))
    doc_topic = model.doc_topic_
    for i in range(10):
        print("{} (top topic: {})".format(titles[i], doc_topic[i].argmax()))

    columns = feature_names
    df_output = pd.DataFrame(topic_word, columns = columns)
    print(df_output.shape)
    df_output.to_csv(r'./data/2_webofsci_LDA_features_{}.csv'.format(data_type))

    vectorized_func = np.vectorize(lambda x: 1 if x > 0.001 else 0)
    df_output = df_output.apply(vectorized_func)
    print(df_output.shape)

    # * Adding TF-IDF features
    if data_type == "train":
        df = df.sample(n=1000, random_state=42).reset_index(drop=True)
    else:
        df = df.sample(n=1000, random_state=42).reset_index(drop=True)
    tf_idf_vectorizer = TfidfVectorizer(vocabulary = feature_names)
    tf_idf = tf_idf_vectorizer.fit_transform(df[0])
    matrix_keywords_words = tf_idf.toarray()
    pd_data = pd.DataFrame(matrix_keywords_words, columns = columns)
    pd_data.to_csv(r'./data/4_webofsci_{}_tf-idf_features.csv'.format(data_type))
    my_final_df = pd_data.append(df_output, ignore_index = True)
    my_final_df.to_csv(r'./data/4_webofsci_{}_LDA_features.csv'.format(data_type))
    print(df_output.shape)