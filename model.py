# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 22:10:40 2017

@author: altescy
"""


import numpy as np
from chainer import Chain, links as L, functions as F
from chainer import Variable

class Seq2SeqAttention(Chain):
    def __init__(self, n_in, n_mid, n_out, ignore_label=-1):
        super(Seq2SeqAttention, self).__init__(
            embedx = L.EmbedID(n_in, n_mid, ignore_label=ignore_label),
            embedy = L.EmbedID(n_out, n_mid, ignore_label=ignore_label),
            lstm_i = L.LSTM(n_mid, n_mid),  # Encoder用LSTM
            lstm_c = L.LSTM(n_mid, n_mid),  # Encoder -> Decoder 変換素子
            lstm_o = L.LSTM(n_mid, n_mid),  # Decoder用LSTM
            w_c    = L.Linear(n_mid, n_mid),
            w_h    = L.Linear(n_mid, n_mid),
            out    = L.Linear(n_mid, n_out)
        )
        self.ignore_label = ignore_label
        self.n_mid = n_mid

    def __call__(self, X, Y):
        self.reset_state()
        p, H = self.encode(X)
        #p = F.dropout(p)
        p = F.dropout(self.lstm_c(p))
        loss = self.decode_train(p, H, Y)
        return loss

    def predict(self, X, max_iter=30, eos=1):
        self.reset_state()
        p, H = self.encode(X)
        return self.decode(self.lstm_c(p), H, n_iter=max_iter, eos=eos)

    def encode(self, X):
        sortedidx = np.argsort([-len(x) for x in X]).astype(np.int32)
        X_T = F.transpose_sequence(X[sortedidx])

        h = self.lstm_i(self.embedx(X_T[0]))
        H = [[np.copy(h_i.data)] for h_i in h]
        for x in X_T[1:]:
            h = self.lstm_i(self.embedx(x))[:x.shape[0]]
            for i, h_i in enumerate(h):
                H[i].append(np.copy(h_i.data))
        H = [np.array(H_i, dtype=np.float32) for H_i in H]

        H = [H[i] for i in np.argsort(sortedidx)]
        h = F.array.permutate.permutate(self.lstm_i.h, np.argsort(sortedidx).astype(np.int32))
        return h, H

    def decoder(self, y_prev, H, n_batch=None, train=True):
        if not n_batch:
            n_batch = len(H)
        h = self.lstm_o(y_prev)[:n_batch]

        c = []
        for H_i, o_i in zip(H, h):
            a_i = np.exp(np.dot(H_i, o_i.data))
            a_i = a_i / np.sum(a_i)
            c.append(np.dot(H_i.T, a_i))
        c = Variable(np.array(c).astype(np.float32))

        q = F.tanh(self.w_h(h) + self.w_c(c))
        return self.out(q)

    def decode_train(self, p, H, Y):
        sortedidx = np.argsort([-len(y) for y in Y]).astype(np.int32)
        Y_T = F.transpose_sequence(Y[sortedidx])
        p = F.array.permutate.permutate(p, sortedidx)
        H = [H[i] for i in sortedidx]

        loss = 0
        y_prev = p  # 最初の入力は(変換素子を通した)Encoderからの出力
        for y in Y_T:
            q = self.decoder(y_prev, H, len(y))
            loss += F.softmax_cross_entropy(q[:len(y)], y)
            y_prev = self.embedy(y)

        return loss

    def decode(self, p, H, n_iter=30, eos=1):
        P = []
        y_prev = p
        for _ in range(n_iter):
            q = self.decoder(y_prev, H, train=False)
            P.append(np.argmax(q.data, axis=1).astype(np.int32))
            y_prev = self.embedy(P[-1])
        P = np.array(P).T

        # 出力用に<eos>以下を切り捨てる
        out = []
        for p in P:
            for j, val in enumerate(p):
                if val == eos:
                    break
            out.append(p[:j+1])
        return np.array(out)

    def reset_state(self):
        self.lstm_i.reset_state()
        self.lstm_c.reset_state()
        self.lstm_o.reset_state()
