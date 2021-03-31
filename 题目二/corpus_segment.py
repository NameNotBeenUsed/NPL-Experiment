# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 09:58:13 2018

@author: lenovo
"""
import os
import jieba
from Tools import writefile,readfile

#将每一段文本转换为分词后的独立文档输出
def corpus_segment(corpus_path,seg_path):
    '''
    raw_path为原始文本路径
    paragraph_path为分词后文本路径
    '''
    catelist = os.listdir(corpus_path)#获取原始路径下的所有子目录
    '''
    题目中先是类别名
    然后是每个文件
    '''
    print('开始分词')
    for mydir in catelist:
        '''mydir为题目下的一个类别'''
        class_path = corpus_path + mydir + '/'
        seg_dir = seg_path + mydir + '/'
        
        if not os.path.exists(seg_dir):
            os.makedirs(seg_dir)
        
        file_list = os.listdir(class_path)#获取未分段语料库中某一类别下的所有文本
        
        for file_path in file_list:
            fullname = class_path + file_path
            content = readfile(fullname)
            
            content_seg = jieba.cut(content)
            writefile(seg_dir + file_path, ' '.join(content_seg))
                
    print('中文语料分词结束')
    
def main():
    corpus_path = './训练集/'
    seg_path = './训练集分词/'
    corpus_segment(corpus_path,seg_path)
    
    corpus_path = './测试集分类/'
    seg_path = './测试集分词/'
    corpus_segment(corpus_path,seg_path)
    
if __name__ == '__main__':
    main()