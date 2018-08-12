#!/usr/bin/python
#-*- coding: utf-8 -*-
#将最原始的数据进行分词 保存到train_data_fenci.txt
#用词典做的，估计效率会比较低

import jieba
import jieba.analyse
import sys
import logging

'''# 该函数未被使用
def cutWords(msg,stopWords):  
    seg_list = jieba.cut(msg,cut_all=False)  
    #key_list = jieba.analyse.extract_tags(msg,20) #get keywords   
    leftWords = []   
    for i in seg_list:  
        if (i not in stopWords):  
            leftWords.append(i)          
    return leftWords  '''

def cutWord(train_file,test_file):
    train_data = open(train_file,encoding='gb18030')
    test_data = open(test_file,encoding='gb18030')   
    stop=loadStopWords()

    train_data_count=0
    test_data_count=0
    ID_dict={}
    Age_dict={}
    Gender_dict={}
    Education_dict={}
    train_keywords_dict = {}
    test_keywords_dict = {}
    train_key_word_list=[]
    test_key_word_list=[]
    test_ID_list=[]

    # 下面的循环的作用是将训练集中的所有的 ID 的关键词提取出来
    for single_query in train_data:
        train_data_count+=1
        # 将特征分割开来
        single_query_list = single_query.split()
        ID  = single_query_list.pop(0)
        ID_dict[ID]=ID
        Age_dict[ID]=single_query_list.pop(0) 
        Gender_dict[ID]=single_query_list.pop(0)
        Education_dict[ID]=single_query_list.pop(0)
        # 如果已经存在该 ID 的记录，那么在原来的基础上继续添加关键词，如果没有，则初始化为空列表
        key_word_list=train_keywords_dict.get(ID,[])
        for j,sentence in enumerate(single_query_list):
        	# 提出每句话的关键词
            train_key_word =  jieba.analyse.extract_tags(sentence)
            # 只有当用户的所有信息都已知的时候，才提取该用户的关键词
            if( Age_dict[ID]!='0'and Gender_dict[ID]!='0' and Education_dict[ID]!=0):
                for i  in train_key_word:
                    if(i not in stop):
                        key_word_list.append(i)
                print ('processing %d in %d'%(j,train_data_count))
        if len(train_keywords_dict.get(ID,[])) == 0:
        	train_keywords_dict[ID] = key_word_list

    # 下面的循环的作用是将测试集中的所有的 ID 的关键词提取出来
    for single_query in test_data:
        test_data_count+=1
        # 将特征分割开来
        single_query_list = single_query.split()
        ID  = single_query_list.pop(0)
        # 如果 ID 存在，那么在原来的列表的基础上继续添加，如果不存在，那么就创建一个新的列表
        test_key_word_list=test_keywords_dict.get(ID,[])
        # key_word_list 和 dic 的 value 指向的是一样的，改变 key_word_list 就是改变 dic 的 value
        for k,sentence in enumerate(single_query_list):
            test_key_word = jieba.analyse.extract_tags(sentence)#基于 TF-IDF 算法的关键词抽取
            for i  in test_key_word:
                if(i not in stop):
                    test_key_word_list.append(i) 
            print ('test_processing %d in %d'%(k,test_data_count))
        if len(test_keywords_dict.get(ID, [])) == 0:
        	test_keywords_dict[ID] = test_key_word_list

    # 将所有信息完整的训练集的用户的信息保存到一个新的文件中
    with open('train_data_fenci.txt','w') as fw_dict_keywords:
          for key,value in train_keywords_dict.items():
            if(Age_dict[key]!='0'and Gender_dict[key]!='0'and Education_dict[key]!='0'):
                fw_dict_keywords.write('{0}'.format(key))
                fw_dict_keywords.write(' '+(Age_dict[key]))
                fw_dict_keywords.write(' '+Gender_dict[key])
                fw_dict_keywords.write(' '+Education_dict[key]+' ')
                fw_dict_keywords.write(' '.join((value))+'\n')
    print ('cutWord file save in train_data_fenci.txt')

    # 将所有测试集的关键词保存到一个新的文件中
    with open('test_data_fenci.txt','w') as fw_dict_keywords:
           for key,value in test_keywords_dict.items():
                fw_dict_keywords.write('{0}'.format(key)+' ')
                fw_dict_keywords.write(' '.join((value))+'\n')  
    print ('cutWord file save in test_data_fenci.txt')  


    train_data.close()
    test_data.close()        


#获取停用词表，将停用词保存到列表中
def loadStopWords():   
    stop = [line.strip()  for line in open('stopwords.txt',encoding='gb18030').readlines() ]   
    return stop  

# 该函数并没有被用到
'''def get_train_table(train_data):
    ID_dict={}
    Age_dict={}
    Gender_dict={}
    Education_dict={}
    feature_dict={}
    keywords_dict = {}
    for single_query in train_data:
        single_query_list = single_query.split()
        ID = single_query_list.pop(0)
        ID_dict[ID]=ID
        Age_dict[ID]=single_query_list.pop(0)
        Gender_dict[ID]=single_query_list.pop(0)
        Education_dict[ID]=single_query_list.pop(0)
    return ID_dict,Age_dict,Gender_dict,Education_dict'''

def main():

    # train_file = 'user_tag_query.2W.TRAIN.csv'
    # test_file = 'user_tag_query.2W.TEST.csv'
    train_file = './data/user_tag_query.2W.TRAIN'
    test_file = './data/user_tag_query.2W.TEST'
    #print(get_train_table(train_data))
    cutWord(train_file,test_file)

    #print(loadStopWords())




if __name__ == '__main__':
    main()