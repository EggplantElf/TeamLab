from __future__ import division
from sentence import *
from model import *
from random import shuffle,choice
import sys
sys.path.append('liblinear/python/')
import liblinearutil as ll


def write_instances(input_file, output_file, extra_output_file, model_file, train = True):
    if train:
        model = Model()
        feat_func = model.register_features
        pos_func = model.register_pos
    else:
        model = Model(model_file)
        feat_func = model.map_features
        pos_func = model.map_pos

    f = open(output_file, 'w')
    g = open(extra_output_file, 'w')

    instances = []
    for sent in read_sentence(input_file):
        for t in sent:
            pos = pos_func(t.gold_pos) # 5
            feat = t.maxent_features(feat_func) # [10, 25, 38, 1067]
            prev_pos1 = t.prev_pos(1)
            prev_pos2 = t.prev_pos(2)
            next_pos1 = t.next_pos(1)
            next_pos2 = t.next_pos(2)

            instances.append((pos, feat, prev_pos1, prev_pos2, next_pos1, next_pos2))

    # register all possible prev_pos and next_pos, 
    # and make sure that the indices are larger than other features
    # to facilitate adding them as extra features
    # if train:
    #     for pos in model.pos_dict:
    #         feat_func('PREV_POS1:%s' % pos)
    #         feat_func('PREV_POS2:%s' % pos)
    #         for pos2 in model.pos_dict:
    #             feat_func('PREV_POS1:%s' % (pos, pos2))



    #     feat_func('PREV_POS1:BOS')
    #     feat_func('PREV_POS2:BOS')
    #     for pos in model.pos_dict:
    #         feat_func('NEXT_POS1:%s' % pos)
    #         feat_func('NEXT_POS2:%s' % pos)
    #     feat_func('NEXT_POS1:EOS')
    #     feat_func('NEXT_POS2:EOS')


    for (pos, feat, prev_pos1, prev_pos2, next_pos1, next_pos2) in instances:
        pos_feat = sorted([feat_func('PREV_POS1:%s' % prev_pos1),\
                           feat_func('PREV_POS2:%s' % prev_pos2),\
                           feat_func('NEXT_POS1:%s' % next_pos1),\
                           feat_func('NEXT_POS2:%s' % next_pos2)])
        f.write('%d %s\n' % (pos, ' '.join('%d:1' % i for i in feat)))
        g.write('%d %s\n' % (pos, ' '.join('%d:1' % i for i in feat + pos_feat)))

    f.close()
    g.close()
    if train:
        model.write_stats('feat.txt')
    model.save(model_file)


def predict(input_file, model0_file, model1_file, dict_file, output_file):
    # out = open(output_file, 'w')
    m0 = ll.load_model(model0_file)
    m1 = ll.load_model(model1_file)
    dicts = Model(dict_file)
    print '# of features:', len(dicts.feature_dict)

    # for easier mapping from neighbouring pos tags to features
    pos2feat = {}
    for pos in dicts.pos_dict_rev: #[0, 1, 2, 3, ...]
        pos2feat[(-1, pos)] = dicts.map_features('PREV_POS1:%s' % dicts.map_pos_rev(pos))
        pos2feat[(-2, pos)] = dicts.map_features('PREV_POS2:%s' % dicts.map_pos_rev(pos))
        pos2feat[(1, pos)] = dicts.map_features('NEXT_POS1:%s' % dicts.map_pos_rev(pos))
        pos2feat[(2, pos)] = dicts.map_features('NEXT_POS2:%s' % dicts.map_pos_rev(pos))

    pos2feat[(-1, 'BOS')] = dicts.map_features('PREV_POS1:BOS')
    pos2feat[(-2, 'BOS')] = dicts.map_features('PREV_POS2:BOS')
    pos2feat[(1, 'EOS')] = dicts.map_features('NEXT_POS1:EOS')
    pos2feat[(2, 'EOS')] = dicts.map_features('NEXT_POS2:EOS')

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

        y_2 = [choice(xrange(1, len(dicts.pos_dict))) for i in y_1]

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
    for i in xrange(200):
        shuffle(order)
        for j in order:
            p1 = pos2feat[(-1, y_[j - 1])] if j > 0 else pos2feat[(-1, 'BOS')]
            p2 = pos2feat[(-2, y_[j - 2])] if j > 1 else pos2feat[(-2, 'BOS')]
            n1 = pos2feat[(1, y_[j + 1])] if j < len(y_) - 1 else pos2feat[(1, 'EOS')]
            n2 = pos2feat[(2, y_[j + 2])] if j < len(y_) - 2 else pos2feat[(2, 'EOS')]


            # cache the prediction of given prev and next pos
            if (p2, p1, n1, n2) in cache[j][1]:
                ny = cache[j][1][(p2, p1, n1, n2)]
            else:
                x = [{k:1 for k in x_[j] + list((p2, p1, n1, n2))}]
                ny = int(ll.predict([], x, model, '-q')[0][0])
                cache[j][1][(p2, p1, n1, n2)] = ny

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
        if same > len(y_):
            break
    # return [max(votes[k], key = lambda x: votes[k][x]) for k in xrange(len(y_))]
    return y_

if __name__ == '__main__':
    if '-i' in sys.argv: 
        write_instances('../data/pos/train.col', 'train.nopos.inst', 'train.pos.inst', 'maxent.dict', True)
        write_instances('../data/pos/dev.col', 'dev.nopos.inst', 'dev.pos.inst', 'maxent.dict', False)
    if '-p' in sys.argv:
        predict('../data/pos/dev.col', 'models/m0.model', 'models/m1.model', 'maxent.dict', 'dev.predict.col')


