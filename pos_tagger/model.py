from __future__ import division
import numpy as np
import itertools as it
import operator as op
import cPickle
from random import random



class Model:
    def __init__(self):
        self.feature_dict = {'#': 0}
        self.pos_dict = {}
        self.pos_dict_rev = {}

    def map_features(self, feature):
        return self.feature_dict.get(feature, None)


    def register_features(self, feature):
        if feature not in self.feature_dict:
            self.feature_dict[feature] = len(self.feature_dict)
        return self.feature_dict[feature]

    def register_pos(self, pos):
        if pos not in self.pos_dict:
            i = len(self.pos_dict)
            self.pos_dict[pos] = i
            self.pos_dict_rev[i] = pos
        return self.pos_dict[pos]

    def map_pos_rev(self, index):
        return self.pos_dict_rev[index]

    def create_weights(self):
        self.featlen = len(self.feature_dict)
        self.poslen = len(self.pos_dict)
        print 'features:', self.featlen
        print 'pos tags:', self.poslen
        self.weights = np.zeros(shape = (len(self.feature_dict), len(self.pos_dict)))
        self.delta = np.zeros(shape = (len(self.feature_dict), len(self.pos_dict)))


    def get_scores(self, features):
        return np.sum(self.weights[i] for i in features)

    def predict(self, scores):
        c, m = None, -999
        for i, p in enumerate(scores):
            if p > m:
                c = i
                m = p
        return c

    # simple perceptron update
    def update(self, f, g, p):
        for i in f:
            self.weights[i, g] += 1
            self.weights[i, p] -= 1


    # passive-aggresive update with average
    def update_pa(self, f, g, p, scores, q = 1, C = 0.1):
        t = min((scores[p] - scores[g] + 1) / (2 * len(f)), C)
        # t = (scores[p] - scores[g] + 1) / (2 * len(f) + 1 / (2 * C))
        for i in f:
            self.weights[i, g] += t
            self.weights[i, p] -= t
            self.delta[i, g] += t * q
            self.delta[i, p] -= t * q

    def average(self, q):
        self.weights -= self.delta / q 


def save(model, filename):
    cPickle.dump(model, open(filename, 'wb'))

def load(filename):
    return cPickle.load(open(filename, 'rb'))
