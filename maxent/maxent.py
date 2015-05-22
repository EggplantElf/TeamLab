from __future__ import division
from sentence import *
from model import *
from random import shuffle,choice
import sys
sys.path.append('liblinear/python/')
import liblinearutil as ll


def write_instances(input_file, output_file, extra_output_file ,model_file, train = True):
    if train:
        model = Model()
        feat_func = model.register_features
        pos_func = model.register_pos
    else:
        model = Model(model_file)
        feat_func = model.map_features
        pos_func = model.map_pos
        print '# of features:', len(model.feature_dict)

    f = open(output_file, 'w')
    g = open(extra_output_file, 'w')

    instances = []
    for sent in read_sentence(input_file):
        for t in sent:
            pos = pos_func(t.gold_pos) # 5
            feat = t.maxent_features(feat_func) # [10, 25, 38, 1067]
            prev_pos = t.prev_pos()
            next_pos = t.next_pos()
            instances.append((pos, feat, prev_pos, next_pos))

    # register all possible prev_pos and next_pos, 
    # and make sure that the indices are larger than other features
    # to facilitate adding them as extra features
    if train:
        for pos in model.pos_dict:
            feat_func('PREV_POS:%s' % pos)
        feat_func('PREV_POS:BOS')
        for pos in model.pos_dict:
            feat_func('NEXT_POS:%s' % pos)
        feat_func('NEXT_POS:EOS')


    for (pos, feat, prev_pos, next_pos) in instances:
        pos_feat = sorted([feat_func('PREV_POS:%s' % prev_pos),\
                           feat_func('NEXT_POS:%s' % next_pos)])
        f.write('%d %s\n' % (pos, ' '.join('%d:1' % i for i in feat)))
        g.write('%d %s\n' % (pos, ' '.join('%d:1' % i for i in feat + pos_feat)))

    f.close()
    g.close()
    model.save(model_file)


def predict(input_file, model0_file, model1_file, dict_file, output_file):
    # out = open(output_file, 'w')
    m0 = ll.load_model(model0_file)
    m1 = ll.load_model(model1_file)
    dicts = Model(dict_file)

    pos2feat = {}
    for pos in dicts.pos_dict_rev: #[0, 1, 2, 3, ...]
        pos2feat[(-1, pos)] = dicts.map_features('PREV_POS:%s' % dicts.map_pos_rev(pos))
        pos2feat[(1, pos)] = dicts.map_features('NEXT_POS:%s' % dicts.map_pos_rev(pos))

    pos2feat[(-1, 'BOS')] = dicts.map_features('PREV_POS:BOS')
    pos2feat[(1, 'EOS')] = dicts.map_features('NEXT_POS:EOS')
       
    s1, s2 = 0, 0
    total = 0

    for sent in read_sentence(input_file):
        x_ = []
        g_ = []
        for t in sent:
            feat = t.maxent_features(dicts.map_features)
            x_.append(feat)
            g_.append(dicts.map_pos(t.gold_pos))
        y_1 = map(int, ll.predict([], [{k:1 for k in f} for f in x_], m0, '-q')[0])

        # y_2 = [choice(xrange(1, len(dicts.pos_dict))) for i in y_1]

        y_2 = inference(m1, pos2feat, y_1[:], x_)

        # print dicts.pos_dict
        for y, y1, y2 in zip(g_, y_1, y_2):
            if y == y1:
                s1 += 1
            if y == y2:
                s2 += 1
            total += 1

    print 'acc 1: %d / %d = %.4f' % (s1, total, s1 / total)
    print 'acc 2: %d / %d = %.4f' % (s2, total, s2 / total)



def inference(model, pos2feat, y_, x_):
    cache = []
    for i in xrange(len(y_)):
        cache.append((x_[i], {}))

    # sampling
    count = 0
    same = 0
    votes = [{} for i in y_]
    order = range(len(y_))
    for i in xrange(100):
        shuffle(order)
        for j in order:
            p = pos2feat[(-1, y_[j - 1])] if j > 0 else pos2feat[(-1, 'BOS')]
            n = pos2feat[(1, y_[j + 1])] if j < len(y_) - 1 else pos2feat[(1, 'EOS')]

            # cache the prediction of given prev and next pos
            if (p, n) in cache[j][1]:
                ny = cache[j][1][(p, n)]
            else:
                x = [{k:1 for k in x_[j] + list((p, n))}]
                ny = int(ll.predict([], x, model, '-q')[0][0])
                cache[j][1][(p, n)] = ny

            if y_[j] == ny:
                same += 1
            else:
                same = 0

            y_[j] = ny

            count += 1
            if count > 200:
                for k, y in enumerate(y_):
                    if y not in votes[k]:
                        votes[k][y] = 1
                    else:
                        votes[k][y] += 1
        if same > 20:
            break
    # return [max(votes[k], key = lambda x: votes[k][x]) for k in xrange(len(y_))]
    return y_

if __name__ == '__main__':
    # write_instances('../data/pos/train.col', 'train.nopos.inst', 'train.pos.inst', 'maxent.dict', True)
    # write_instances('../data/pos/dev.col', 'dev.nopos.inst', 'dev.pos.inst', 'maxent.dict', False)
    predict('../data/pos/dev.col', 'm0.model', 'm1.model', 'maxent.dict', 'dev.predict.col')

