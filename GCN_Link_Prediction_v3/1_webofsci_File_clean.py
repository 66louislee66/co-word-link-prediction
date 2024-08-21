# 清洗web of science中的文件，保留摘要部分
# ? 导入模块
from nltk.corpus import stopwords  #从nltk库中导入stopwords模块
import nltk  # 导入nltk库
from utils import clean_str,get_wordnet_pos,is_valid
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

dataset = ["train", "test"]
for data_type in dataset:
    
    # ? 初步清洗，只保留摘要部分
    with open(r'/home/lym/lab/project_work/project_versions/GCN_Link_Prediction_v3/data/webofsci_{}.txt'.format(data_type), "r", encoding='UTF-8') as input_file, open(r'/home/lym/lab/project_work/project_versions/GCN_Link_Prediction_v3/data/webofsci_{}_clean.txt'.format(data_type), "w+", encoding='UTF-8') as output_file:
        flag = False
        for line in input_file:
            if line.startswith('AB '):
                flag = True
                output_file.write(line[3:-1])
            elif flag and line.startswith('   '):
                output_file.write(line[2:-1])
            elif flag and not line.startswith('   '):
                output_file.write('\n')
                flag = False

    # ? 创建文档内容列表doc_content_list
    nltk.download('stopwords')  
    stop_words = set(stopwords.words('english'))  
    print(stop_words)  

    doc_content_list = []  
    f = open(r'/home/lym/lab/project_work/project_versions/GCN_Link_Prediction_v3/data/webofsci_{}_clean.txt'.format(data_type), 'rb') 
    for line in f.readlines():
        doc_content_list.append(line.strip().decode('latin1')) 
    f.close()

    # ? 分词（词干提取、词型还原）
    # * 参考：https://www.cnblogs.com/cpaulyz/p/13717637.html
    wordReduction_docs = []  # 存储所有词性还原的词
    for sentence in doc_content_list:
        temp = clean_str(sentence)
        tokens = word_tokenize(temp)  # 分词
        tagged_sent = pos_tag(tokens)  # 获取单词词性

        wnl = WordNetLemmatizer() # 创建还原词性的对象
        lemmas_sent = []
        for tag in tagged_sent:  # 遍历词性列表
            wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN  # 获取词性的标记
            lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))  # 词形还原
        lemmas_str = ' '.join(lemmas_sent).strip()
        wordReduction_docs.append(lemmas_str)

    # ? 过滤
    allclean_docs = []  # 存储最终清理的文档单词
    for sentence in wordReduction_docs:
        temp = clean_str(sentence)
        tokens = word_tokenize(temp)
        # 过滤无效单词
        filtered_tokens = [word for word in tokens if is_valid(word)]
        allclean_str = ' '.join(filtered_tokens).strip()
        allclean_docs.append(allclean_str)

    # ? 创建词频字典
    word_freq = {}  # 存储单词和频率的键值对

    for doc_content in allclean_docs: 
        temp = clean_str(doc_content)  
        words = temp.split() 
        for word in words: 
            if word in word_freq: 
                word_freq[word] += 1 
            else:
                word_freq[word] = 1  

    # ? 取高频词组成词表
    clean_docs = []  
    for doc_content in allclean_docs:  
        temp = clean_str(doc_content) 
        words = temp.split() 
        doc_words = []  
        for word in words:  
            if word not in stop_words and word_freq[word] >= 5: 
                doc_words.append(word)  

        doc_str = ' '.join(doc_words).strip()  
        clean_docs.append(doc_str) 

    clean_corpus_str = '\n'.join(clean_docs)  

    f = open(r'/home/lym/lab/project_work/project_versions/GCN_Link_Prediction_v3/data/webofsci_{}_allclean.txt'.format(data_type), 'w') 
    f.write(clean_corpus_str)  
    f.close()

    # ? 求文档列表最小单词数、最大单词数及平均单词数
    min_len = 10000  
    aver_len = 0  
    max_len = 0  

    f = open(r'/home/lym/lab/project_work/project_versions/GCN_Link_Prediction_v3/data/webofsci_{}_allclean.txt'.format(data_type), 'r') 
    lines = f.readlines()  
    for line in lines:  
        line = line.strip() 
        temp = line.split() 
        aver_len = aver_len + len(temp) 
        if len(temp) < min_len:  
            min_len = len(temp) 
        if len(temp) > max_len: 
            max_len = len(temp) 
    f.close()
    aver_len = 1.0 * aver_len / len(lines)  
    print('min_len : ' + str(min_len))
    print('max_len : ' + str(max_len))
    print('average_len : ' + str(aver_len))