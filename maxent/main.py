from __future__ import division
from sentence import *
from model import *
from random import shuffle
import sys
import cProfile
import itertools as it
import numpy as np

def train(train_file, model_file):
    model = Model()
    instances = []
    feat_func = model.register_features


    # register features for model 0
    for sent in read_sentence(train_file):
        for t in sent:
            pos = model.register_pos(t.gold_pos) # 5
            f = t.maxent_features(model.register_features) # [10, 25, 38, 1067]
            p1 = t.prev_pos(1)
            p2 = t.prev_pos(2)
            n1 = t.next_pos(1)
            n2 = t.next_pos(2)
            instances.append((pos, f, f[:], p1, p2, n1, n2))
    model.create_weights(0)

    model.register_pos('BOS')
    model.register_pos('EOS')

    # register features for model 1
    for (pos, f0, f1, p1, p2, n1, n2) in instances:
        pos_feat = [feat_func('POS_P1:%s' % p1),\
                    feat_func('POS_P2:%s' % p2),\
                    feat_func('POS_N1:%s' % n1),\
                    feat_func('POS_N2:%s' % n2),\
                    feat_func('POS_P1_P2:%s_%s' % (p1, p2)),\
                    feat_func('POS_N1_N2:%s_%s' % (n1, n2)),\
                    feat_func('POS_P1_N1:%s_%s' % (p1, n1)),\
                    ]
        # f1 += filter(lambda x: x != None, pos_feat)
        f1 += pos_feat
    model.create_weights(1)

    print 'instances:', len(instances)

    # train model 0
    q = 0
    for i in range(20):
        total = 0
        correct = 0
        for (g, f0, f1, p1, p2, n1, n2) in instances:
            scores = model.get_scores(0, f0)
            p = model.predict(scores)
            if p != g:
                model.update_pa(0, f0, g, p, scores, q, 1)
            else:
                correct += 1
            total += 1
            q += 1
        print 'iteration %d done, accuracy: %.4f' % (i+1, correct / total)
    model.average(0, q)

    # train model 1
    q = 0
    for i in range(20):
        total = 0
        correct = 0
        for (g, f0, f1, p1, p2, n1, n2) in instances:
            scores = model.get_scores(1, f1)
            p = model.predict(scores)
            if p != g:
                model.update_pa(1, f1, g, p, scores, q, 1)
            else:
                correct += 1
            total += 1
            q += 1
        print 'iteration %d done, accuracy: %.4f' % (i+1, correct / total)
    model.average(1, q)



    model.write_stats('feat.txt')
    model.save(model_file)
    print 'done training'
    return model

def predict(input_file, model_file, output_file):
    global bos, eos
    out = open(output_file, 'w')
    model = Model(model_file)
    bos = model.map_pos('BOS')
    eos = model.map_pos('EOS')
    print '# of features:', len(model.feat_dict)

    # for easier mapping from neighbouring pos tags to features
    p_f = {}
    for i in model.pos_dict_rev: #[0, 1, 2, 3, ...] bos and eos are also included
        pi = model.map_pos_rev(i)
        p_f[(-1, i)] = model.map_features('POS_P1:%s' % pi)
        p_f[(-2, i)] = model.map_features('POS_P2:%s' % pi)
        p_f[(+1, i)] = model.map_features('POS_N1:%s' % pi)
        p_f[(+2, i)] = model.map_features('POS_N2:%s' % pi)

        for j in model.pos_dict_rev:
            pj = model.map_pos_rev(j)
            p_f[(-1, -2, i, j)] = model.map_features('POS_P1_P2:%s_%s' % (pi, pj))
            p_f[(+1, +2, i, j)] = model.map_features('POS_N1_N2:%s_%s' % (pi, pj))
            p_f[(-1, +1, i, j)] = model.map_features('POS_P1_N1:%s_%s' % (pi, pj))

    s0, s1 = 0, 0
    total = 0

    for sent in read_sentence(input_file):
        x_ = []
        y_0 = []
        g_ = []
        for t in sent:
            f0 = t.maxent_features(model.map_features)
            scores = model.get_scores(0, f0)
            y0 = model.predict(scores)
            x_.append(f0)
            y_0.append(y0)
            g_.append(model.map_pos(t.gold_pos))
        y_1 = inference(model, p_f, y_0[:], x_, propose_deterministic)
        for t, g, y0, y1 in zip(sent, g_, y_0, y_1):
            if y0 == g:
                s0 += 1
            if y1 == g:
                s1 += 1
            total += 1
            out.write('%s\t%s\n' % (t.word, model.map_pos_rev(y1)))
        out.write('\n')
    out.close()
    print 'acc 0: %d / %d = %.4f' % (s0, total, s0 / total)
    print 'acc 1: %d / %d = %.4f' % (s1, total, s1 / total)


def inference(model, p_f, y_, x_, propose_func):
    cache = [{} for y in y_]
    count = 0
    same = 0
    votes = [{} for i in y_]
    order = range(len(y_))
    for i in xrange(200):
        shuffle(order)
        for j in order:
            feats = pos_feat(p_f, y_, j)
            ny = propose_func(model, cache, feats, x_, j)

            if y_[j] == ny:
                same += 1
            else:
                same = 0

            y_[j] = ny

            count += 1
            if count > 100:
                for k, y in enumerate(y_):
                    if y not in votes[k]:
                        votes[k][y] = 1
                    else:
                        votes[k][y] += 1
        if count > 200 and same > 2 * len(y_):
            break
    # return [max(votes[k], key = lambda x: votes[k][x]) for k in xrange(len(y_))]
    return y_

def propose_deterministic(model, cache, feats, x_, j):
    if feats in cache[j]:
        ny = cache[j][feats]
    else:
        f1 = x_[j] + list(feats)
        scores = model.get_scores(1, f1)
        ny = model.predict(scores)
        cache[j][feats] = ny
    return ny

def pos_feat(p_f, y_, j):
    p1 = y_[j - 1] if j > 0 else bos
    p2 = y_[j - 2] if j > 1 else bos
    n1 = y_[j + 1] if j < len(y_) - 1 else eos
    n2 = y_[j + 2] if j < len(y_) - 2 else eos
    feats = map(p_f.__getitem__, [(-1, p1), (-2, p2), (1, n1), (2, n2), \
                                  (-1, -2, p1, p2), (1, 2, n1, n2), (-1, 1, p1, n1)])
    return tuple(filter(lambda x: x != None, feats))




def stats(model_file):
    model = Model(model_file)
    model.zeros()


if __name__ == '__main__':
    train('../data/pos/train.col', 'new.model')
    predict('../data/pos/dev.col', 'new.model', 'predict.col')
    stats('new.model')