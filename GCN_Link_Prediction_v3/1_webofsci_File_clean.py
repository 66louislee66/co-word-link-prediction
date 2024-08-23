# 清洗web of science中的文件，保留摘要部分
# * 导入模块
from utils import clean_str,is_valid
from nltk import word_tokenize


dataset = ["train", "test"]
for data_type in dataset:
    
    # * 初步清洗，只保留摘要部分，原始文档
    with open(r'./data/1_webofsci_{}.txt'.format(data_type), "r", encoding='UTF-8') as input_file, open(r'./data/1_webofsci_{}_clean.txt'.format(data_type), "w+", encoding='UTF-8') as output_file:
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

    # * 创建文档内容列表doc_content_list
    doc_content_list = []  
    f = open(r'./data/1_webofsci_{}_clean.txt'.format(data_type), 'rb') 
    for line in f.readlines():
        doc_content_list.append(line.strip().decode('latin1')) 
    f.close()
    
    # * 自定义停用词
    resource_path = r'./StopwordsList/stopwords_en.txt'
    with open(resource_path, encoding='utf-8') as stop_fil:
        stop_words = set(stop_fil.read().lower().split("\n"))

    # * 分词与过滤
    # ! 参考：https://www.cnblogs.com/cpaulyz/p/13717637.html
    
    allclean_docs = []  # 存储最终清理的文档单词
    for sentence in doc_content_list:
        temp = clean_str(sentence)
        tokens = word_tokenize(temp)
        filtered_tokens = [word for word in tokens if is_valid(word)]
        allclean_str = ' '.join(filtered_tokens).strip()
        allclean_docs.append(allclean_str)

    # * 去停用词
    clean_docs = []  
    for doc_content in allclean_docs:  
        temp = clean_str(doc_content) 
        words = temp.split() 
        doc_words = []  
        for word in words:  
            if word not in stop_words: 
                doc_words.append(word)  

        doc_str = ' '.join(doc_words).strip()  
        clean_docs.append(doc_str) 

    clean_corpus_str = '\n'.join(clean_docs)  

    f = open(r'./data/1_webofsci_{}_allclean.txt'.format(data_type), 'w') 
    f.write(clean_corpus_str)  
    f.close()

    # * 求文档列表最小单词数、最大单词数及平均单词数
    min_len = 10000  
    aver_len = 0  
    max_len = 0  

    f = open(r'./data/1_webofsci_{}_allclean.txt'.format(data_type), 'r') 
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