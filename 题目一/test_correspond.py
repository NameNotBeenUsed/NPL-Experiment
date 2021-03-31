# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 14:10:27 2018

@author: lenovo
"""

import os
from bs4 import BeautifulSoup
from Tools import writefile
from sklearn.datasets.base import Bunch

def find_index(L,e):
    return [i for (i,j) in enumerate(L) if j == e]

'''
匹配测试集中每个样本的正确标签
分放至不同类别文件夹下
'''
def correspond(corpus_path, para_path, test_path, classify_path):
    '''
    corpus_path为语料库路径
    para_path为语料库分段存储后的路径
    test_path为未经分类的训练集路径
    classify_path为训练集分类后的存储路径
    '''
    catelist = os.listdir(corpus_path)
    bunch_corpus = Bunch(target_name=[], label=[], filenames=[], contents=[])
    bunch_corpus.target_name.extend(catelist)
    '''分别将语料库与测试集读入两个列表'''
    '''读入语料库列表'''
    print('开始分段')
    for mydir in catelist:
        class_path = corpus_path + mydir + '/'
        para_dir = para_path + mydir + '/'
        if not os.path.exists(para_dir):
            os.makedirs(para_dir)
            
        file_list = os.listdir(class_path)
        
        for file_path in file_list:
            fullname = class_path + file_path
            soup = BeautifulSoup(open(fullname,encoding='utf8'),'lxml')
            contenttitles = soup.find_all('contenttitle')
            contents = soup.find_all('content')
            for contenttitle,content,i in zip(contenttitles,contents,range(0,len(contenttitles),1)):
                sentence = contenttitle.get_text() + '。' + content.get_text()
                writefile(para_dir + mydir + str(i) + '.txt',sentence)
                bunch_corpus.label.append(mydir)
                bunch_corpus.filenames.append(para_dir + mydir + str(i) + '.txt')
                bunch_corpus.contents.append(sentence)
    
    bunch_test = Bunch(label=[], content=[])           
    soup_test = BeautifulSoup(open(test_path,encoding = 'utf8'),'lxml')
    contenttitles = soup_test.find_all('contenttitle')
    contents = soup_test.find_all('content')
    for contenttitle,content in zip(contenttitles,contents):
        bunch_test.content.append(contenttitle.get_text() + '。' + content.get_text())
        
    '''匹配'''
    for test_passage in bunch_test.content:
        index = bunch_corpus.contents.index(test_passage)
        bunch_test.label.append(bunch_corpus.label[index])
        
    '''测试集分类写入'''
    for mydir in catelist:
        classify_dir = classify_path + mydir + '/'
        if not os.path.exists(classify_dir):
            os.makedirs(classify_dir)
            
        corre = find_index(bunch_test.label,mydir)
        for index,i in zip(corre,range(0,len(corre),1)):
            writefile(classify_dir + mydir + str(i) + '.txt',bunch_test.content[index])
             
corpus_path = './搜狐2012新闻语料切分（utf8）/'
para_path = './语料库/'
test_path = 'test.html'
classify_path = './测试集分类/'
correspond(corpus_path, para_path, test_path, classify_path)