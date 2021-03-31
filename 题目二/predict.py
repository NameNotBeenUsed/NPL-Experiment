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
#from sklearn.model_selection import GridSearchCV
#from minepy import MINE
#import numpy as np

trainpath = "训练集词袋/tfdifspace.dat"
train_set = readbunchobj(trainpath)

testpath = "测试集词袋/testspace.dat"
test_set = readbunchobj(testpath)

def metrics_result(actual, predict):
    print('精度:{0:.3f}'.format(metrics.precision_score(actual, predict, average='weighted')))
    print('召回:{0:0.3f}'.format(metrics.recall_score(actual, predict, average='weighted')))
    print('f1-score:{0:.3f}'.format(metrics.f1_score(actual, predict, average='weighted')))

'''print('*************NB*************\n')
clf = MultinomialNB(alpha=0.001).fit(train_set.tdm, train_set.label)

predicted = clf.predict(test_set.tdm)

for flabel, file_name, expct_cate in zip(test_set.label, test_set.filenames, predicted):
    if flabel != expct_cate:
        print(file_name, ": 实际类别:", flabel, " -->预测类别:", expct_cate)
        
print("预测完毕")  
metrics_result(test_set.label, predicted)'''

print('*************SVM*************\n')
clf = SVC()
clf.fit(train_set.tdm,train_set.label)

predicted = clf.predict(test_set.tdm)

for flabel,file_name,expct_cate in zip(test_set.label,test_set.filenames,predicted):
    if flabel != expct_cate:
        print(file_name,'：实际类别：',flabel,'-->预测类别：',expct_cate)
        
print('预测完毕')
metrics_result(test_set.label,predicted)

'''print('*************NB&chi2*************\n')
ch2 = SelectKBest(chi2, k=10000)
train_x = ch2.fit_transform(train_set.tdm, train_set.label)
test_x = ch2.transform(test_set.tdm)

#clf = SVC(kernel = 'sigmoid')
#clf = SVC()
#clf.fit(train_x,train_set.label)
clf = MultinomialNB(alpha=0.001).fit(train_x, train_set.label)
predicted = clf.predict(test_x)

for flabel,file_name,expct_cate in zip(test_set.label,test_set.filenames,predicted):
    if flabel != expct_cate:
        print(file_name,'：实际类别：',flabel,'-->预测类别：',expct_cate)
        
print('预测完毕')
metrics_result(test_set.label,predicted)'''

'''print('*************SVM_GRID*************\n')
#parameters = {'kernel':('linear','rbf'),'C':[1,2,4],'gamma':[0.125,0.25,0.5,1,2,4]}
parameters = {'C':[1,2,4]}
svr = SVC()
clf = GridSearchCV(svr,parameters,n_jobs = -1)
clf.fit(train_set.tdm,train_set.label)

predicted = clf.predict(test_set.tdm)

for flabel,file_name,expct_cate in zip(test_set.label,test_set.filenames,predicted):
    if flabel != expct_cate:
        print(file_name,'：实际类别：',flabel,'-->预测类别：',expct_cate)
        
print('预测完毕')
metrics_result(test_set.label,predicted)'''

'''print('*************Mutual Information*************\n')
def mic(x,y):
    m = MINE()
    m.compute_score(x,y)
    return (m.mic(),0.5)
    
mi = SelectKBest(lambda X, Y:np.fromiter(map(lambda x:mic(x, Y), X.T),dtype = np.float).T, k=2)
train_x = mi.fit_transform(train_set.tdm, train_set.label)
text_x = mi.transform(test_set.tdm)
clf = MultinomialNB(alpha=0.001).fit(train_x, train_set.label)
predicted = clf.predict(text_x)

for flabel,file_name,expct_cate in zip(test_set.label,test_set.filenames,predicted):
    if flabel != expct_cate:
        print(file_name,'：实际类别：',flabel,'-->预测类别：',expct_cate)
        
print('预测完毕')
metrics_result(test_set.label,predicted)'''