# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 22:26:35 2017

@author: altescy
"""


from flask import Flask, render_template, request
import pickle
import numpy as np
from chainer import Chain, links as L, functions as F
from chainer import Variable

import sys
sys.path.append('/app/')
from model import Seq2SeqAttention
from utilities.word2id import Word2ID


def generate(titles, modelfile, max_len=50):
    with open('./data/KADOKAWA-caption-id-dataset.pkl', 'rb') as f:
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


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def generate_abstract():
    if request.method == 'POST':
        title = request.form['title']
        abst = generate([title], \
                        './model/seq2seq-Full-atteention-300epochs.mdl')[0]
        return render_template('index.html', title=title, abst=abst)
    return render_template('index.html')




if __name__ == '__main__':
    app.run()
