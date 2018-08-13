# coding: utf-8
import sys
from time import time
from sklearn import preprocessing, linear_model
from scipy.sparse import bmat
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import HashingVectorizer,CountVectorizer,TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV 
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np
# olderr = np.seterr(all='ignore') 



# 从整个训练集数据集中抽取部分数据作为训练模型的训练集数据和测试集数据，并且指定要使用的目标变量
def input_data(train_file,divide_number,end_number,tags):
    train_words = []
    train_tags=[]
    test_words = []
    test_tags=[]
    with open(train_file, 'r',encoding='gb18030') as f:
        text=f.readlines()

        # 构建训练集数据
        train_data=text[:divide_number]   
        for single_query in train_data:
        	# 先将所有的字段分割
            single_query_list = single_query.split(' ')
            # 去除 ID 字段
            single_query_list.pop(0)#id
            # 标签不确定的情况下构建样本
            if(single_query_list[tags]!='0'):
            	# 构建训练集样本的目标变量
                train_tags.append(single_query_list[tags])
                # 删除3个目标变量，剩下关键词
                single_query_list.pop(0)
                single_query_list.pop(0)
                single_query_list.pop(0)
                # 列表转换为字符串，列表中的逗号转换为空格，将字符串所有的单引号去掉，将列表的左中括号和右中括号去掉，最后将换行符去掉
                # 所以最后剩下的是所有关键词构成的字符串，它们分别用逗号分隔
                train_words.append((str(single_query_list)).replace(',',' ').replace('\'','').lstrip('[').rstrip(']').replace('\\n',''))

        #构建测试集数据，构建的方法和训练集数据的构建是一样的
        test_data=text[divide_number:end_number]   
        for single_query in test_data:
            single_query_list = single_query.split(' ')
            single_query_list.pop(0)#id
            if(single_query_list[tags]!='0'):
                test_tags.append(single_query_list[tags])
                single_query_list.pop(0)
                single_query_list.pop(0)
                single_query_list.pop(0)
                test_words.append((str(single_query_list)).replace(',',' ').replace('\'','').lstrip('[').rstrip(']').replace('\\n',''))
       # print(test_words)
        #print(test_tags_age)
    print('input_data done!')

    # 返回构建的训练集输入，训练集目标变量，测试集输入，测试集目标变量
    return train_words, train_tags, test_words, test_tags

 
# 对训练集文件和测试集文件进行处理，产生训练集输入、训练集目标变量、测试集输入
# 这一步是在我们训练好模型之后，要对比赛的数据进行预测的时候才进行的
# 当然，我们这里并不是比赛，所以这一个函数对我们来说并没有什么用
# 但是我们可以了解一下作者的思路，想一想我们在比赛的时候会怎么做
'''def input_data_write_tags(train_file, test_file,tags):
    train_words = []
    train_tags=[]
    test_words = []

    with open(train_file, 'r',encoding='gb18030') as f:
        text=f.readlines()
        train_data=text[0:]   
        for single_query in train_data:
            single_query_list = single_query.split(' ')
            single_query_list.pop(0)
            train_tags.append(single_query_list[tags])
            single_query_list.pop(0)
            single_query_list.pop(0)
            single_query_list.pop(0)
            train_words.append((str(single_query_list)).replace(',',' ').replace('\'','').lstrip('[').rstrip(']').replace('\\n',''))

    with open(test_file, 'r') as f:
        text=f.readlines()
        test_data=text[0:]
        for single_query in test_data:
            single_query_list = single_query.split(' ')
            single_query_list.pop(0)
            test_words.append((str(single_query_list)).replace(',',' ').replace('\'','').lstrip('[').rstrip(']').replace('\\n',''))
            
   
    print('input_data done!')
    return train_words, train_tags,test_words'''

 
'''def write_test_tags(test_file,test_tags_age,test_tags_gender,test_tags_education):
    test_ID=[]
    with open(test_file,'r') as test_data:
        for single_query in test_data:
            single_query_list=single_query.split(' ')
            test_ID.append(single_query_list[0])

    with open('test_tags_file_cv_hv_chi2_59000_15000_13500.csv','w',encoding='gbk') as test_tags_file:
        for x in range(0,len(test_tags_age)):
            test_tags_file.write(test_ID[x]+' '+test_tags_age[x]+' '+test_tags_gender[x]+' '+test_tags_education[x]+'\n')  '''
  

# 简单的词频统计，包含的信息少，不建议使用，本项目中使用的是 tf-idf
'''def vectorize(train_words,test_words,n_feature):
    print ('*************************HashingVectorizer*************************')  
    v = HashingVectorizer(n_features=n_feature,non_negative =True)
    print("n_features:%d"%n_feature)
    train_data = v.fit_transform(train_words)
    test_data = v.fit_transform(test_words)
    print ("the shape of train is "+repr(train_data.shape))
    print ("the shape of test is "+repr(test_data.shape)) 
    
     
    return train_data, test_data
    #print('vectorize done!')'''
    
    
# 将训练集数据和测试集数据转换为 tf-idf 特征矩阵，然后使用 chi2 进行特征选择
def tfidf_vectorize_1(train_words, train_tags, test_words, n_dimensionality):
    #method 2:TfidfVectorizer  
    print ('*************************\nTfidfVectorizer\n*************************')   
    # sublinear_tf 用于对 tf 进行缩放，例如把 tf 替换成 1 + log(tf)
    # 一般情况下我们是这么做的，而不是直接使用 tf（也就是关键词在文档中出现的次数）
    # 其实可以在这里设置 stop_words，而不用在前面分词的时候处理
    # 现在的问题是如何设置中文的停用词
    tv = TfidfVectorizer(sublinear_tf = True)
                                          
    tfidf_train_2 = tv.fit_transform(train_words);  #得到矩阵
    tfidf_test_2 = tv.transform(test_words)
    print ("the shape of train is "+repr(tfidf_train_2.shape))  
    print ("the shape of test is "+repr(tfidf_test_2.shape))
    train_data,test_data=feature_selection_chi2(tfidf_train_2,train_tags,tfidf_test_2,n_dimensionality) 
    return  train_data, test_data


'''def tfidf_vectorize(train_words,train_tags,test_words,test_tags,n_dimensionality):
    #method 2:TfidfVectorizer  
    print ('*************************TfidfVectorizer+chi2*************************')   
    t0=time()
    tv = TfidfVectorizer(sublinear_tf = True) # 
                                          
    tfidf_train_2 = tv.fit_transform(train_words);  #得到矩阵
    # tv2 = TfidfVectorizer(vocabulary = tv.vocabulary_);  
    tfidf_test_2 = tv.transform(test_words);  
    print ("the shape of train is "+repr(tfidf_train_2.shape))  
    print ("the shape of test is "+repr(tfidf_test_2.shape))
    train_data,test_data=feature_selection_chi2(tfidf_train_2,train_tags,tfidf_test_2,n_dimensionality) 
    print("done in %0.3fs." % (time() - t0))
    return  train_data,test_data'''


# hash vectorie 提供的信息较少，所以并不建议使用这种方法。
'''def feature_union_tv_hv(train_words,train_tags,test_words,test_tags,n_feature,n_dimensionality):
    print('*************************feature_union_tv_hv*************************')
    hv = HashingVectorizer(n_features=n_feature,non_negative =True)
    tv1 = TfidfVectorizer(sublinear_tf = True,  max_df = 0.5) #
    train_combined_features = FeatureUnion([('hv',hv),('tv1',tv1)])
    train_data=train_combined_features.fit_transform(train_words)
    print ("the shape of train is "+repr(train_data.shape))  
    tv2 = TfidfVectorizer(vocabulary = tv1.vocabulary_ ) 
    test_combined_features = FeatureUnion([('hv',hv),('tv2',tv2)])  
    test_data=test_combined_features.fit_transform(test_words)
    print ("the shape of train is "+repr(test_data.shape)) 
    train_data,test_data=feature_selection_chi2(train_data,train_tags,test_data,n_dimensionality) 
    return train_data,test_data'''


# 使用两种方式进行特征提取，然后将这两种方式的结果合并起来
# 第一种方式是使用 TFIDF 提取特征，然后进行 LDA 降维
# 第二种方式是使用 TFIDF 提取特征，然后使用 chi2 进行特征选择
def feature_union_lda_tv(train_words,test_words,train_tags,n_dimensionality,n_topics):
 	# LDA主题提取
    print('*************************feature_union_lda_tv*************************')
    train_data_lda,test_data_lda = LDA(train_words,test_words,n_topics)
    # 归一化lda
    train_data_lda_normalize=preprocessing.normalize(train_data_lda, norm='l2')
    test_data_lda_normalize=preprocessing.normalize(test_data_lda, norm='l2')
    # #向量化
    train_data_tv,test_data_tv = tfidf_vectorize_1(train_words,train_tags,test_words,n_dimensionality)  
    #特征矩阵合并
    train_data=bmat([[train_data_lda_normalize, train_data_tv]])
    test_data=bmat([[test_data_lda_normalize, test_data_tv]])
    return train_data,test_data

    
# 函数为被使用
'''def sgd_single(train_data,test_data,train_tags):
    print ('*************************\nSVM\n*************************' )
    clf = linear_model.SGDClassifier()
    clf.fit(train_data,train_tags)  
    pred_tags = clf.predict(test_data) 
    print('clf done!')
    return pred_tags    
'''
    
# 使用支持向量机来进行分类
# 先用训练集数据训练模型，然后对测试集数据进行预测
def SVM_single(train_data,test_data,train_tags): 
#SVM Classifier  
    print ('******************************SVM*****************************' )
    t0=time()
    # 这里我们可以设置不同的 kernel，看看它们各自的效果
    svclf = SVC(kernel = 'linear')#default with 'rbf'  
    svclf.fit(train_data,train_tags)  
    pred_tags = svclf.predict(test_data) 
    print("done in %0.3fs." % (time() - t0))
    print('clf done!')
    return pred_tags


# 使用 accuracy 来评估模型的性能，这里有多个类别，所以使用 accuracy 来评估模型的性能不是太好
# 问题1：AUC、ROC、F1 这样的性能评估指标是否能应用到多分类的应用中？
# 问题2：如果可以，那么怎么写代码来实现？（可以直接调用工具包）
def evaluate_single(test_tags, test_tags_prediction):
    actual=test_tags
    pred=test_tags_prediction
    print ('accuracy_score:{0:.3f}'.format(accuracy_score(actual, pred)))
    print('confusion_matrix:')
    print(confusion_matrix(actual, pred))


# 使用 chi2 方法来选择 n_dimensionality 个最重要的特征
def feature_selection_chi2(train_data,train_tags,test_data,n_dimensionality):

    print('feature_selection_chi2'+'\n'+'n_dimensionality:%d' %n_dimensionality)
    ch2= SelectKBest(score_func=chi2, k=n_dimensionality)
    train_data=ch2.fit_transform(train_data,train_tags)
    test_data=ch2.transform(test_data)
    return train_data  , test_data 


# 先用 TFIDF 构建特征矩阵，然后使用 lda 对特征矩阵进行降维
def LDA(train_words,test_words,n_topics):
    print("Extracting tf features for LDA...")
    # max_df 和 min_df 决定了忽略什么词语
    # max_df：如果是小数，那么表示的是百分比，当词语出现的文档超过这个百分比的时候，就忽略这个词语，它相当于是停用词；如果是整数，那么就表示具体的文档的数量
    # min_df：如果是小数，那么表示的是百分比，当词语出现的文档低于这个百分比的时候，就忽略这个词语；如果是整数，那么就表示具体的文档的数量
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2)
    t0 = time()
    train_tf = tf_vectorizer.fit_transform(train_words)
    test_tf = tf_vectorizer.transform(test_words)
    print("done in %0.3fs." % (time() - t0))
    # n_components 参数和 n_topics 参数是一样的，n_topics 参数在以后会被抛弃
    # learning_method 可以设置为 online 或者为 batch，当数据量很大的时候，使用 online 方法会比较快
    lda = LatentDirichletAllocation(n_components=n_topics,max_iter=10,learning_method='online')
    t0 = time()
    print('n_topics:%d' %n_topics)
    train_word_lda=lda.fit_transform(train_tf)
    test_word_lda=lda.transform(test_tf)
    print(" done in %0.3fs." % (time() - t0))
    return  train_word_lda,test_word_lda


# 分别对3个目标变量进行测试
def test():
    #            标签（年龄性别学历）  卡方选取后的维数    主题个数
    # 0 对应 age
    # 1 对应 Gender
    # 2 对应 Education
    # 我们分别以这三个标签作为我们的目标变量进行训练
    test_single(0,59500,100)
    test_single(1,12000,5)
    test_single(2,130,10)


# 对指定的目标变量（age，Gender，Education）进行测试
# 在最后的实现中，n_dimensionality 和 n_topics 都没有被用到
# 但是它们应该被用到，因为处理后的数据的特征的数量太大了，这样所需要的训练时间会非常长
# 因此，我们应该对特征进行适当处理，例如特征选择、降维等等
def test_single(tags,n_dimensionality,n_topics):
    train_file = 'train_data_fenci.txt'
    # 测试集起始样本位置
    divide_number=15500
    # 测试集终止样本位置
    end_number=17633#17633
    # n_feature=320000
    print('file:'+train_file)
    print('tags:%d   ' % tags )
    # tag="age"
    #将数据分为训练与测试，获取训练与测试数据的标签
    train_words, train_tags, test_words, test_tags = input_data(train_file,divide_number,end_number,tags)
    # 方法一：tv + 卡方选择，选择指定数量的最重要的那些特征
    # train_data,test_data= tfidf_vectorize_1(train_words, train_tags, test_words, n_dimensionality)
 	# 方法二：tv + 卡方选择，tv + LDA，然后进行特征融合
    train_data,test_data=feature_union_lda_tv(train_words,test_words,train_tags,n_dimensionality,n_topics)
    
    test_tags_prediction=SVM_single(train_data,test_data,train_tags)
    #计算正确率
    evaluate_single(np.asarray(test_tags), test_tags_prediction)


# 下面的函数有错误：函数中并没有创建 test_tags 这个变量，但是竟然调用了这个变量
'''def write_single(train_file,test_file,tags,n_dimensionality,n_topics):
    n_feature=320000
    train_words, train_tags, test_words = input_data_write_tags(train_file, test_file,tags)
    # #向量化
    # #hv
    #train_data,test_data = vectorize(train_words,test_words,n_feature)
    # #tv
    train_data,test_data= tfidf_vectorize(train_words,train_tags,test_words,test_tags,n_dimensionality)
 	#lda_tv
    #train_data,test_data=feature_union_lda_tv(train_words,test_words,train_tags,test_tags,n_dimensionality,n_topics)
    #tv_hv 
    #train_data,test_data = feature_union_tv_hv(train_words,train_tags,test_words,test_tags,n_feature,n_dimensionality)
    
    test_tags_pre=SVM_single(train_data,test_data,train_tags)
    return test_tags_pre'''


# 对测试集数据进行预测，并且将预测结果保存到文件中
'''def write():
    train_file='train_data_fenci.txt'
    test_file = 'test_data_fenci.txt'
    test_tags_age_pred=write_single(train_file,test_file,0,59000,50)
    test_tags_gender_pred=write_single_1(train_file,test_file,1,15000,50)
    test_tags_education_pred=write_single(train_file,test_file,2,135000,50)
    write_test_tags(test_file,test_tags_age_pred,test_tags_gender_pred,test_tags_education_pred)'''


#########################################
def optimize_single(tags):
    train_file = 'train_data_fenci.txt'
    devide_number=15000
    end_number=17633#17633
    # n_feature=1000
    print('file:'+train_file)
    print('tags:%d   ' %tags )
    train_words, train_tags,test_words, test_tags = input_data(train_file,devide_number,end_number,tags)
    train_data,test_data = tfidf_vectorize_1(train_words, train_tags, test_words, n_dimensionality)

    pipeline = Pipeline([     
    #('TfidfVectorizer',TfidfVectorizer(sublinear_tf = True)),
    ('feature_selection',SelectKBest(chi2)),
    ('clf',SGDClassifier()), 
    ]);

    a=np.linspace(10000,200000,num=1000,dtype=int)
   # min_df=np.linspace(0,0.01,num=1000,dtype=float)

    parameters={
    #'TfidfVectorizer__sublinear_tf':[True,False],
   # 'TfidfVectorizer__min_df':list(min_df),
    #'TfidfVectorizer__sublinear_tf':[True,False],

    #'feature_selection__score_func':[chi2],
   'feature_selection__k':list(a)
    }
    grid_search = GridSearchCV(pipeline,parameters,n_jobs =6,verbose=1);  
    print("Performing grid search...")  
    print("pipeline:", [name for name, _ in pipeline.steps])  
    print("parameters:")  
    #pprint(parameters)  
  
    grid_search.fit(train_words, train_tags)  
    print("Best score: %0.3f" % grid_search.best_score_)  
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


def optimize():
    optimize_single(0)
    optimize_single(1)
    optimize_single(2)


def main():
    # 如果第一个参数是 test，那么对3个目标变量分别进行测试，看看分类效果如何
    if(sys.argv[1]=="test"):
        test()
    if(sys.argv[1]=="optimize"):
   		optimize()

if __name__ == '__main__':
    main()
