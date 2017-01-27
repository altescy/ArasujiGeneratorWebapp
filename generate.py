# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 22:07:53 2017

@author: altescy
"""


import pickle
import numpy as np

from model import Seq2SeqAttention
from utilities.word2id import Word2ID


def generate(titles, modelfile, max_len=50):
    with open('data/KADOKAWA-caption-id-dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)

    t_wd2id = Word2ID()
    c_wd2id = Word2ID()
    t_wd2id.deserialize(dataset['title'])
    c_wd2id.deserialize(dataset['caption'])
    
    with open(modelfile, 'rb') as f:
        modeldata = pickle.load(f)

    model = modeldata['model']
    title_id = t_wd2id(titles)
    for i in range(len(title_id)):
        title_id[i].append(-1)
    title_id = np.array([np.array(d, dtype=np.int32) for d in title_id])
        
    prediction = c_wd2id.translate(model.predict(title_id, max_iter=max_len))
    
    return [p.replace(' ', '').replace('<eos>', '') for p in prediction]



if __name__ == '__main__':
    title =['この素晴らしい世界に祝福を！２',
            'あいまいみー ～Surgical Friends～',
            '小林さんちのメイドラゴン',
            'Fate／Grand Order ‐First Order‐',
            '幼女戦記',
            '亜人ちゃんは語りたい',
            'ガウリールドロップアウト',
            'けものフレンズ',
            '学戦都市アスタリスク'
            ]
    
    modelfile = 'model/model-i/seq2seq-Full-atteention-300epochs.mdl'
    P = generate(title, modelfile)
    for i, p in enumerate(P):
        print('『'+title[i]+'』:\n    ', p)