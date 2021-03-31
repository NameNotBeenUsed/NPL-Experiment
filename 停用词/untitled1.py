# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 19:44:39 2018

@author: lenovo
"""
import codecs

def readfile(raw_file_path):  
    with codecs.open(raw_file_path,"r",encoding="gb2312") as f:  
        raw_file=f.readlines() 
    return raw_file

def writefile(file_path,file):
    with codecs.open(file_path,'w','utf8') as f:
        f.write(file)

       
stopword_path = "stop_words.txt"
doc = []
f = readfile(stopword_path)
for sentence in f:
    sentence = sentence.strip()
    doc.append(sentence)
writefile('1.txt','\n'.join(doc))