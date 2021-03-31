# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 19:05:35 2018

@author: lenovo
"""

from sklearn.datasets.base import Bunch
from sklearn.feature_extraction.text import TfidfVectorizer
from Tools import readfile,readbunchobj,writebunchobj

def vector_space(stopword_path,bunch_path,space_path,train_tfidf_path=None):
    stpwrdlst = readfile(stopword_path).splitlines()
    bunch = readbunchobj(bunch_path)
    tfidfspace = Bunch(target_name=bunch.target_name, 
                       label=bunch.label, 
                       filenames=bunch.filenames, 
                       tdm=[],
                       vocabulary={})
    if train_tfidf_path is not None:
        trainbunch = readbunchobj(train_tfidf_path)
        tfidfspace.vocabulary = trainbunch.vocabulary
        vectorizer = TfidfVectorizer(stop_words=stpwrdlst, 
                                     sublinear_tf=True, 
                                     max_df=0.5,
                                     vocabulary=trainbunch.vocabulary)
        tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)
        
    else:
        vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True, max_df=0.5)
        tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)
        tfidfspace.vocabulary = vectorizer.vocabulary_#A mapping of terms to feature indices.
        
    writebunchobj(space_path, tfidfspace)
    print("tf-idf词向量空间实例创建成功")
    
def main():
    stopword_path = "训练集词袋/stop_words.txt"
    bunch_path = "训练集词袋/训练集.dat"
    space_path = "训练集词袋/tfdifspace.dat"
    vector_space(stopword_path, bunch_path, space_path)

    bunch_path = "测试集词袋/测试集.dat"
    space_path = "测试集词袋/testspace.dat"
    train_tfidf_path = "训练集词袋/tfdifspace.dat"
    vector_space(stopword_path, bunch_path, space_path, train_tfidf_path)
    
if __name__ == '__main__':
    main()
