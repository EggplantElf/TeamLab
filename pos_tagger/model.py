from __future__ import division
import numpy as np
import itertools as it
import operator as op
import cPickle
import gzip
from random import random
from math import sqrt


class Model:
    def __init__(self, modelfile = None):
        if modelfile:
            self.load(modelfile)
        else:
            self.feature_dict = {'#': 0}
            self.pos_dict = {}
            self.pos_dict_rev = {}

    def save(self, modelfile):
        stream = gzip.open(modelfile,'wb')
        cPickle.dump(self.weights,stream,-1)
        cPickle.dump(self.feature_dict, stream, -1)
        cPickle.dump(self.pos_dict, stream, -1)
        cPickle.dump(self.pos_dict_rev, stream, -1)
        stream.close()

    def load(self, modelfile):
        stream = gzip.open(modelfile,'rb')
        self.weights = cPickle.load(stream)
        self.feature_dict = cPickle.load(stream)
        self.pos_dict = cPickle.load(stream)
        self.pos_dict_rev = cPickle.load(stream)
        stream.close()

# TODO feature weights statistics and reduce unsignificant features
    def drop_zeros(self):
        pass


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

    def get_scores_for_dev(self, features):
        return np.sum(self.avg_weights[i] for i in features)

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
        # t = 1
        t = min((scores[p] - scores[g] + 1) / (2 * len(f)), C)
        # t = (scores[p] - scores[g] + 1) / (2 * len(f) + 1 / (2 * C))
        for i in f:
            self.weights[i, g] += t
            self.weights[i, p] -= t
            self.delta[i, g] += t * q
            self.delta[i, p] -= t * q

    def average(self, q):
        self.weights -= self.delta / q 

    def average_for_dev(self, q):
        self.avg_weights = self.weights - (self.delta / q)

    def standard_deviation(self):
        avg = np.sum(self.weights ** 2) / len(self.feature_dict) ** 2
        sd = sqrt(np.sum((self.weights - avg) ** 2) / len(self.feature_dict) ** 2)
        return avg, sd

    def walk(self, d):
        diff = np.random.normal(0, d, self.weights.shape)
        self.new_weights = self.weights + diff

    def get_scores_from_new_weights(self, features):
        return np.sum(self.new_weights[i] for i in features)

    def accept_new_weights(self):
        self.weights = self.new_weights

