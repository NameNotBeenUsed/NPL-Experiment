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
        vectorizer = TfidfVectorizer(stop_words=stpwrdlst, #If a list, that list is assumed to contain stop words, all of which will be removed from the resulting tokens. Only applies if analyzer == 'word'
                                     sublinear_tf=True,#亚线性（sublinear），用于描述量与量之间的一种变化关系，例如y=a+b*x^n，其中0<n<1。当n=1时，表示为线性关系，当n>1时，表示为超线性关系。 
                                     max_df=0.5,#This parameter is ignored if vocabulary is not None.
                                     vocabulary=trainbunch.vocabulary)#Either a Mapping (e.g., a dict) where keys are terms and values are indices in the feature matrix, or an iterable over terms. If not given, a vocabulary is determined from the input documents.
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
