# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 03:27:15 2017

@author: altescy
"""


import json
import pickle
import word2id


if __name__ == '__main__':
    infile = '../data/KADOKAWA-caption-dataset.json'
    outfile = '../data/KADOKAWA-caption-id-dataset.pkl'
    
    with open(infile, 'r') as f:
        dataset = json.load(f)
    
    t_wd2id = word2id.Word2ID()
    c_wd2id = word2id.Word2ID()
    for idx, item in sorted(dataset['dataset'].items()):
        t_wd2id.append([item['title']])
        c_wd2id.append([item['caption']])
    
    
    #t_wd2id.padding()
    #c_wd2id.padding()
        
    dataset_ = {'title'  : t_wd2id.serialize(),
                'caption': c_wd2id.serialize()}
    
    with open(outfile, 'wb') as f:
        pickle.dump(dataset_, f)