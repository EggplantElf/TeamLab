from __future__ import division
import numpy as np
import itertools as it
import cPickle
import gzip
from random import random
from math import sqrt


class Model:
    def __init__(self, modelfile = None):
        if modelfile:
            self.load(modelfile)
        else:
            self.feat_dict = {'#': 0}
            self.pos_dict = {'#': 0}
            self.pos_dict_rev = {0 : '#'}
            self.stats = {'#': 0}
            self.weights = {}
            self.delta = {}


    def save(self, modelfile):
        stream = gzip.open(modelfile,'wb')
        cPickle.dump(self.weights,stream,-1)
        cPickle.dump(self.feat_dict, stream, -1)
        cPickle.dump(self.pos_dict, stream, -1)
        cPickle.dump(self.pos_dict_rev, stream, -1)
        stream.close()

    def load(self, modelfile):
        stream = gzip.open(modelfile,'rb')
        self.weights = cPickle.load(stream)
        self.feat_dict = cPickle.load(stream)
        self.pos_dict = cPickle.load(stream)
        self.pos_dict_rev = cPickle.load(stream)
        stream.close()

    def write_stats(self, feat_file):
        o = open(feat_file, 'w')
        for k in sorted(self.stats):
            l = len(k)
            o.write('%s%s%d  \n' % (k, ' '* (30 -l), self.stats[k]))
        o.close()

    def register_features(self, feature):
        if feature not in self.feat_dict:
            self.feat_dict[feature] = len(self.feat_dict)
            self.stats[feature] = 0
        self.stats[feature] += 1
        return self.feat_dict[feature]

    def register_pos(self, pos):
        if pos not in self.pos_dict:
            i = len(self.pos_dict)
            self.pos_dict[pos] = i
            self.pos_dict_rev[i] = pos
        return self.pos_dict[pos]

    def map_features(self, feature):
        return self.feat_dict.get(feature, None)

    def map_pos(self, pos):
        return self.pos_dict.get(pos, None)

    def map_pos_rev(self, index):
        return self.pos_dict_rev.get(index, None)


    def create_weights(self, m):
        self.weights[m] = np.zeros(shape = (len(self.feat_dict), len(self.pos_dict)))
        self.delta[m] = np.zeros(shape = (len(self.feat_dict), len(self.pos_dict)))


    def get_scores(self, m, features):
        return np.sum(self.weights[m][i] for i in features)

    # def get_scores_for_dev(self, features):
    #     return np.sum(self.avg_weights[i] for i in features)

    def predict(self, scores):
        c, s = None, -999
        for i, p in enumerate(scores):
            if p > s:
                c = i
                s = p
        return c

    # simple perceptron update
    def update(self, m, f, g, p):
        for i in f:
            self.weights[m][i, g] += 1
            self.weights[m][i, p] -= 1


    # passive-aggresive update with average
    def update_pa(self, m, f, g, p, scores, q = 1, C = 0.1):
        t = min((scores[p] - scores[g] + 1) / (2 * len(f)), C)
        # t = (scores[p] - scores[g] + 1) / (2 * len(f) + 1 / (2 * C))
        for i in f:
            self.weights[m][i, g] += t
            self.weights[m][i, p] -= t
            self.delta[m][i, g] += t * q
            self.delta[m][i, p] -= t * q

    def average(self, m, q):
        self.weights[m] -= self.delta[m] / q 


    def zeros(self):
        count = {}
        for w in self.weights[0]:
            i = 1
            s = sum(w ** 2)
            if s == 0:
                if 0 not in count:
                    count[0] = 0
                count[0] += 1
            else:
                while s < i:
                    i /= 10
                if i not in count:
                    count[i] = 0
                count[i] += 1

        for s in sorted(count):
            print s, count[s]

    # def drop_zeros(self, limit = 0.00001):
    #     mapping = {}
    #     for f in self.feat_dict:
    #         i = self.feat_dict[f]
    #         if sum(self.weights[0][i] ** 2) < limit && sum(self.weights[1][i] ** 2) < limit:
    #             j = len(mapping)
    #             mapping[j] = i
    #     new_feat_dict = {}
    #     new_weights = {}
    #     new_weights[0] = np.zeros(shape = (len(mapping), len(self.pos_dict)))
    #     new_weights[1] = np.zeros(shape = (len(mapping), len(self.pos_dict)))




