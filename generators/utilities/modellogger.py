# -*- coding:utf-8 -*-

import pickle

class ModelLogger:
    def __init__(self):
        self.title = self.model = self.epoch = self.errors = self.note = None
    
    def logging(self, title, model, epochs, errors=None, note=None):
        self.title = title
        self.model = model
        self.epochs = epochs
        self.errors = errors
        self.note = note
    
    def serialize(self, title, model, epochs, errors=None, note=None):
        data = {'model' : self.model,
                'epochs': self.epochs,
                'errors': self.errors,
                'note'  : self.note}
        return data
    
    def deserialize(self, data):
        self.model = data['model']
        self.epochs = data['epochs']
        self.errors = data['errors']
        self.note = data['note']        

    def dump(self, title, model, epochs, errors=None, note=None):
        self.logging(title, model, epochs, errors, note)
        data = self.serialize(title, model, epochs, errors, note)
        with open(title+'-%depochs'%epochs, 'wb') as f:
            pickle.dump(data, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        self.deserialize(data)
        self.title = filename.split('.')[0]