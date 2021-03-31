# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 20:46:08 2018

@author: lenovo
"""

from lxml import html
import codecs
import pickle
  
#去除HTML标签
def html2txt(path):  
    with open(path,"rb") as f:  
        content=f.read()
    page = html.document_fromstring(content) # 解析文件  
    text = page.text_content() # 去除所有标签  
    return text  

#写入文件
def writefile(file_path,file):
    with codecs.open(file_path,'w','utf8') as f:
        f.write(file)
        #print('文件写入完毕')

#读出文件   
def readfile(raw_file_path):  
    with codecs.open(raw_file_path,"r",encoding="utf8") as f:  
        raw_file=f.read()  
    return raw_file

def writebunchobj(path,bunchobj):
    with open(path,'wb') as file_obj:
        pickle.dump(bunchobj,file_obj)
        
def readbunchobj(path):
    with open(path,'rb') as file_obj:
        bunch = pickle.load(file_obj)
    return bunch