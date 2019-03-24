# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 11:37:13 2018

@author: s.sikand
"""

import re

def clean_str(string):
    '''
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data
    '''
    string = re.sub(r"[^가-힣A-Za-z0-9(),!?:;.\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r":", " : ", string)
    string = re.sub(r";", " ; ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r'\W+', ' ', string)
    string = string.lower()
    return string.strip()