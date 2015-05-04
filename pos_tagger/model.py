from __future__ import division
import numpy as np
import itertools as it
import operator as op
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
        # used to register previous pos tags before creating the weight matrix
        # for p in self.pos_dict:
        #     for i in range(1, 3):
        #         self.register_features('PREV_POS_%i:%s' % (i, p))

        self.featlen = len(self.feature_dict)
        self.poslen = len(self.pos_dict)
        print 'features:', self.featlen
        print 'pos tags:', self.poslen
        self.weights = np.zeros(shape = (len(self.feature_dict), len(self.pos_dict)))
        self.delta = np.zeros(shape = (len(self.feature_dict), len(self.pos_dict)))

        # self.weights = [ [0.0 for i in xrange(len(self.pos_dict))] for j in xrange(len(self.feature_dict))]

    # def extra_features(self, prev_tags):
    #     for i in range(1, 3):
    #         self.map_features('PREV_POS_%i:%s' % (i, p))


    def get_scores(self, features):
        return np.sum(self.weights[i] for i in features)
        # s = np.zeros(len(self.pos_dict))
        # for i in features:
        #     s += self.weights[i]
        # return s

    # def get_dist(self, scores):
    #     e = np.exp(scores)
    #     return e / sum(e)

    # def predict_stoch(self, dist):
    #     x = random()
    #     for c, p in enumerate(dist):
    #         x -= p
    #         if x < 0:
    #             return c

    def predict(self, scores):
        c, m = None, -999
        for i, p in enumerate(scores):
            if p > m:
                c = i
                m = p
        return c
        # return max(enumerate(scores), key = (lambda x: x[1]))[0] # slower

    # MIRA update with average
    def update(self, f, g, p, scores, q = 1, C = 0.1):
        t = min((scores[p] - scores[g] + 1) / (2 * len(f)), C)
        # t = (scores[p] - scores[g] + 1) / (2 * len(f) + 1 / (2 * C))
        # t = 1
        for i in f:
            self.weights[i, g] += t
            self.weights[i, p] -= t
            self.delta[i, g] += t * q
            self.delta[i, p] -= t * q

    def average(self, q):
        self.weights -= self.delta / q 

