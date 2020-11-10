import matplotlib.pyplot as plt
import string
import jieba
import codecs
import os
from numpy import *
import numpy as np
import pandas as pd
from stop_words import get_stop_words #停用词库
from sklearn.feature_extraction.text import CountVectorizer #用于去掉停用词
from wordcloud import WordCloud #词云
from sklearn import naive_bayes as bayes #朴素贝叶斯分类器
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC

textPath = r"./Naive Bayes/SMSSpamCollection.txt"
with open(textPath,encoding='utf-8') as f:
    txt_list = f.readlines()
ham_mails=[]
spam_mails=[]
ham_prefix='ham	'
spam_prefix='spam	'
for line in txt_list:
    if ham_prefix == line[:len(ham_prefix)]:
        ham_mails.append(line[len(ham_prefix):])
    elif spam_prefix == line[:len(spam_prefix)]:
        spam_mails.append(line[len(spam_prefix):])
    else:
        raise ValueError("格式错误!")
print("正常邮件有",len(ham_mails))
print("垃圾邮件有",len(spam_mails))

#加载停用词库
stop_words = get_stop_words('english')+['...','..']
# print(stop_words,'\n')

# 去停用词处理方法：words_list是字符串列表，filter_words是要过滤掉的单词列表
def words_filter(words_list,filter_words):
    res=[]
    for words in words_list:
        word_list = jieba.cut(words)
        #取小写，过滤掉1个字符长度，以及停用词
        L= filter(lambda w:len(w)>1 and (w not in filter_words), map(str.lower,word_list) )
        res.append( ' '.join(list(L)))
    return res

print(ham_mails[:3]) #去停用词前
ham_mails=words_filter(ham_mails,stop_words)    #去停用词
print(ham_mails[:3]) #去停用词后
spam_prefix=words_filter(spam_prefix,stop_words)    #去停用词

def showWordCloud(text,name):
    wc = WordCloud(
        background_color = "white",
        max_words = 200,
        min_font_size = 15,
        max_font_size = 50,
        width = 400
    )
    wordcloud = wc.generate(text)
    wordcloud.to_file(name+'.png')

# showWordCloud(' '.join(ham_mails),"ham_mails_words")   #正常邮件的词云
# showWordCloud(' '.join(spam_mails),"spam_mails_words")   #垃圾邮件的词云

def transformTextToSparseMatrix(texts):
    vectorizer = CountVectorizer(binary=False)
    vectorizer.fit(texts) # 生成词汇表
    vocabulary = vectorizer.vocabulary_ # 输出词汇表
    vector = vectorizer.transform(texts) # 生成向量
#     print(vector.toarray())
    result = pd.DataFrame(vector.toarray())

    keys = []
    values = []
    for key,value in vectorizer.vocabulary_.items():
        keys.append(key)
        values.append(value)
    df = pd.DataFrame(data={"key":keys, "value": values})
    colnames = df.sort_values("value")["key"].values
    result.columns = colnames
    return result

#将每封邮件转化成向量
data = []
data.extend(ham_mails)
data.extend(spam_mails)

textMatrix = transformTextToSparseMatrix(data)
# textMatrix.head() #展示一下这个稀疏矩阵

features = pd.DataFrame(textMatrix.apply(sum,axis=0))
print(features) #展示总词频数
#抽取出在至少5封邮件都出现的词来作为特征，以减少特征维度
extractedfeatures = [features.index[i] for i in range(features.shape[0]) if features.iloc[i,0]>5]   #得到的是一个下标序列
textMatrix = textMatrix[extractedfeatures]
textMatrix = textMatrix[extractedfeatures]
labels = []
labels.extend(ones(len(ham_mails)))
labels.extend(zeros(len(spam_mails)))
# 划分训练集和测试集
train,test,trainlabel,testlabel = train_test_split(textMatrix,labels,test_size=0.1)


BYM_lb = bayes.BernoulliNB(alpha=1,binarize=True)   #sklearn的朴素贝叶斯分类器
model = BYM_lb.fit(train,trainlabel)    # 调用贝叶斯库函数求解模型

print(train)
print("识别准确率=",model.score(test,testlabel))

