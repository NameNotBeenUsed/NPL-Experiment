# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import pickle

from sklearn.datasets.base import Bunch
from Tools import readfile

'''训练集构建Bunch'''
def corpus2Bunch(wordbag_path,seg_path):
    catelist = os.listdir(seg_path)#类别文件夹名称
    
    bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
    bunch.target_name.extend(catelist)
    
    for mydir in catelist: 
        class_path = seg_path + mydir + '/'
        #每个类别中所有文本文件
        file_list = os.listdir(class_path)
        for file_path in file_list:
            fullname = class_path + file_path
            bunch.label.append(mydir)
            bunch.filenames.append(fullname)
            bunch.contents.append(readfile(fullname))
            
    with open(wordbag_path,'wb') as file_obj:
        pickle.dump(bunch,file_obj)
    print('构建文本对象结束')

def main():
    wordbag_path = '训练集词袋/训练集.dat'
    seg_path = '训练集分词/'
    corpus2Bunch(wordbag_path,seg_path)
    
    wordbag_path = '测试集词袋/测试集.dat'
    seg_path = '测试集分词/'
    corpus2Bunch(wordbag_path,seg_path)
    
if __name__ == '__main__':
    main()