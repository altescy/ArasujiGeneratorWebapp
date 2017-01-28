# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 22:26:35 2017

@author: altescy
"""


from flask import Flask, render_template, request, session, redirect, url_for


from model import Seq2SeqAttention
from generate import generate
from utilities.word2id import Word2ID


app = Flask(__name__)
app.secret_key = 'secret_key'

@app.route('/')
def index():
    if 'title' in session:
        del session['title']
    return render_template('index.html')

@app.route('/post', methods=['POST'])
def generate_abstract():
    session['title'] = request.form['title']
    return redirect(url_for('result'))

@app.route('/result')
def result():
    if 'title' not in session:
        return redirect('')
    title = session['title']
    abst = generate([title], 'model/s2smodel.npz')[0]
    return render_template('index.html', title=title, abst=abst)




if __name__ == '__main__':
    app.run()
