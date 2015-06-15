from __future__ import division
from sentence import *
from model import *
from random import shuffle
import sys
import cProfile
import itertools as it
import random
from time import time
import numpy as np

def train(train_file, mask):
    model = Model()
    instances = []
    for sent in read_sentence(train_file):
        for t in sent:
            f = t.extract_features(model.register_features, mask)
            instances.append((f, model.register_pos(t.gold_pos)))
    model.create_weights()
    print 'instances:', len(instances)

    q = 0
    best_accuracy = 0
    for i in range(10):
        total = 0
        correct = 0
        # shuffle(instances)
        for (f, g) in instances:
            scores = model.get_scores(f)
            p = model.predict(scores)
            if p != g:
                model.update_pa(f, g, p, scores, q, 0.1)
            else:
                correct += 1
            total += 1
            q += 1
        print 'iteration %d done, accuracy: %.4f' % (i+1, correct / total)
        
    model.average(q)
    model.save('test.model')
    print 'done training'
    return model

def read_dev_instances(dev_file):
    instances = []
    for sent in read_sentence(dev_file):
        for t in sent:
            f = t.extract_features(model.map_features, mask)
            instances.append((f, model.register_pos(t.gold_pos)))
    return instances


def evaluate(instances, model, mask, new = False):
    total = 0
    correct = 0
    for (f, g) in instances:
        if new:
            scores = model.get_scores_from_new_weights(f)
        else:
            scores = model.get_scores(f)
        p = model.predict(scores)
        if p == g:
            correct += 1
        total += 1
    return correct / total


def optimize(model, dev_file, mask, num = 1):
    instances = read_dev_instances(dev_file)
    acc = evaluate(instances, model, mask)
    print 'iteration 0: %.4f' % acc
    # for i in range(num):
    #     avg, sd = model.standard_deviation()
    #     print avg, sd
    #     model.walk(0.01)
    #     new_acc = evaluate(instances, model, mask, True)
    #     print 'iteration %d: %.4f' % (i + 1, new_acc)
    #     if new_acc > acc:
    #         model.accept_new_weights()
    #         acc = new_acc



def normalzie(model, dev_file, mask, sigma = 0):
    instances = read_dev_instances(dev_file)
    w = model.weights
    new_model = Model()
    new_model.feature_dict = model.feature_dict
    new_model.pos_dict = model.pos_dict
    new_model.pos_dict_rev = model.pos_dict_rev

    new_model.weights = 1 / np.sqrt(1 / (w + 0.0001) ** 2 + sigma) * np.sign(w)
    acc = evaluate(instances, new_model, mask)
    return acc
    


if __name__ == '__main__':
    # mask = [True] * 7 + [False] * 13
    mask = '11010110111010010010'
    model = train('../data/pos/train.col', mask)
    model = Model('test.model')
    instances = read_dev_instances('../data/pos/dev.col')

    evaluate(instances, model, mask)

    # optimize(model, '../data/pos/train.col', mask, 30)
    # for i in [0, 1, 0.1, 0.01, 0.001, 0.0001]:
    #     print 'sigma =', i
    #     print normalzie(model, '../data/pos/dev.col', mask, i)


