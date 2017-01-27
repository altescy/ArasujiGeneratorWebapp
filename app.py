# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 22:26:35 2017

@author: altescy
"""


from flask import Flask, render_template, request


from model import Seq2SeqAttention
from generate import generate
from utilities.word2id import Word2ID


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def generate_abstract():
    if request.method == 'POST':
        title = request.form['title']
        abst = generate([title], 'model/s2smodel.npz')[0]
        return render_template('index.html', title=title, abst=abst)
    return render_template('index.html')




if __name__ == '__main__':
    app.run()
