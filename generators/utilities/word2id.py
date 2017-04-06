# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 04:41:52 2017

@author: altescy
"""

import pickle

class Word2ID:
    """
    Word2ID
        単語列または文字列から，各単語・文字に固有のID番号を割り当てる
    """
    def __init__(self, source=None):
        """
        input: source
                単語列または文字列の二次元配列
                 - 単語列 source = [['<word>', '<word>', '<word>', ...],
                                   ['<word>', '<word>', '<word>', ...], ...]
                 - 文字列 source = ['< ---- sentence ---- >', 
                                   '< ---- sentence ---- >', ...]
        """
        self.data = []
        self.wd2id = {'<eos>': 1, '<unk>': 0, '<pad>': -1}
        self.id2wd = {1: '<eos>', 0: '<unk>', -1: '<pad>'}
        self.maxlen = 0
        
        if source is not None:
            self.assignID(source)

    
    def __call__(self, source):
        ret = []
        for sentence in source:
            sid = []
            for w in sentence:
                if w in self.wd2id:
                    sid.append(self.wd2id[w])
                else:
                    sid.append(self.wd2id['<unk>'])
            ret.append(sid)
        return ret

    
    def assignID(self, source):
        for sentence in source:           
            sid = []
            for word in sentence:
                if not word in self.wd2id:
                    idx = len(self.wd2id) - 1
                    self.wd2id[word] = idx
                    self.id2wd[idx] = word
                sid.append(self.wd2id[word])
            sid.append(self.wd2id['<eos>'])
            self.data.append(sid)
            
            if len(self.data[-1]) > self.maxlen:
                self.maxlen = len(self.data[-1])
    
    
    def translate(self, source, delim=' ', showpad=False):
        ret = []
        for sentence in source:
            swd = ''
            for i in sentence:
                if self.id2wd[i] == '<pad>':
                    if showpad:
                        swd += self.id2wd[i] + delim
                else:
                    swd += self.id2wd[i] + delim
                
            ret.append(swd)
        return ret
            
    
    def padding(self):
        for idx, sentence in enumerate(self.data):
            d = self.maxlen - len(sentence)
            if d > 0:
                for _ in range(d):
                    self.data[idx].append(self.wd2id['<pad>'])
        
    
    def append(self, source):
        self.assignID(source)
        
    
    def serialize(self):
        return {'data'  : self.data,
                'wd2id' : self.wd2id,
                'id2wd' : self.id2wd,
                'maxlen': self.maxlen}
    

    def deserialize(self, data):
        self.data   = data['data']
        self.wd2id  = data['wd2id']
        self.id2wd  = data['id2wd']
        self.maxlen = data['maxlen']
        
    
    def dump(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.serialize(), f)
    
    
    def load(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.deserialize(data)
    