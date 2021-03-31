# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 19:52:14 2018

@author: lenovo
"""
from sklearn.naive_bayes import MultinomialNB  # 导入多项式贝叶斯算法
from sklearn import metrics
from Tools import readbunchobj
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2 

trainpath = "训练集词袋/tfdifspace.dat"
train_set = readbunchobj(trainpath)

testpath = "测试集词袋/testspace.dat"
test_set = readbunchobj(testpath)

def metrics_result(actual, predict):
    print('精度:{0:.3f}'.format(metrics.precision_score(actual, predict, average='weighted')))
    print('召回:{0:0.3f}'.format(metrics.recall_score(actual, predict, average='weighted')))
    print('f1-score:{0:.3f}'.format(metrics.f1_score(actual, predict, average='weighted')))
    
print('*************NB&chi2*************\n')
ch2  = SelectKBest(chi2, k=10000)
train_x = ch2.fit_transform(train_set.tdm, train_set.label)
test_x = ch2.transform(test_set.tdm)

clf = MultinomialNB(alpha=0.001).fit(train_set.tdm, train_set.label)
clf.fit(train_x,train_set.label)
predicted = clf.predict(test_x)

for flabel,file_name,expct_cate in zip(test_set.label,test_set.filenames,predicted):
    if flabel != expct_cate:
        print(file_name,'：实际类别：',flabel,'-->预测类别：',expct_cate)
        
print('预测完毕')
metrics_result(test_set.label,predicted)

print('*************NB*************\n')
clf = MultinomialNB(alpha=0.001).fit(train_set.tdm, train_set.label)

predicted = clf.predict(test_set.tdm)

for flabel, file_name, expct_cate in zip(test_set.label, test_set.filenames, predicted):
    if flabel != expct_cate:
        print(file_name, ": 实际类别:", flabel, " -->预测类别:", expct_cate)
        
print("预测完毕")    
metrics_result(test_set.label, predicted)

print('*************SVM(poly)*************\n')
#clf = SVC(kernel = 'linear')
clf = SVC(kernel = 'poly')
clf.fit(train_set.tdm,train_set.label)

predicted = clf.predict(test_set.tdm)

for flabel,file_name,expct_cate in zip(test_set.label,test_set.filenames,predicted):
    if flabel != expct_cate:
        print(file_name,'：实际类别：',flabel,'-->预测类别：',expct_cate)
        
print('预测完毕')
metrics_result(test_set.label,predicted)

print('*************CHI*************\n')
ch2  = SelectKBest(chi2, k=10000)
train_x = ch2.fit_transform(train_set.tdm, train_set.label)
test_x = ch2.transform(test_set.tdm)

clf = SVC(kernel = 'linear')
clf.fit(train_x,train_set.label)
predicted = clf.predict(test_x)

for flabel,file_name,expct_cate in zip(test_set.label,test_set.filenames,predicted):
    if flabel != expct_cate:
        print(file_name,'：实际类别：',flabel,'-->预测类别：',expct_cate)
        
print('预测完毕')
metrics_result(test_set.label,predicted)










