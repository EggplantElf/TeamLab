from __future__ import division
from sentence import *
from mapping import *
from random import shuffle,choice
import sys
sys.path.append('liblinear/python/')
import liblinearutil as ll


def write_instances(input_file, output_file, extra_output_file, mapping_file, train = True):
    if train:
        mapping = Mapping()
        feat_func = mapping.register_features
        pos_func = mapping.register_pos
    else:
        mapping = Mapping(mapping_file)
        feat_func = mapping.map_features
        pos_func = mapping.map_pos

    f = open(output_file, 'w')
    g = open(extra_output_file, 'w')

    instances = []
    for sent in read_sentence(input_file):
        for t in sent:
            pos = pos_func(t.gold_pos) # 5
            feat = t.maxent_features(feat_func) # [10, 25, 38, 1067]
            p1 = t.prev_pos(1)
            p2 = t.prev_pos(2)
            n1 = t.next_pos(1)
            n2 = t.next_pos(2)
            instances.append((pos, feat, p1, p2, n1, n2))

    pos_func('BOS')
    pos_func('EOS')

    for (pos, feat, p1, p2, n1, n2) in instances:
        pos_feat =[feat_func('POS_P1:%s' % p1),\
                    feat_func('POS_P2:%s' % p2),\
                    feat_func('POS_N1:%s' % n1),\
                    feat_func('POS_N2:%s' % n2),\
                    feat_func('POS_P1_P2:%s_%s' % (p1, p2)),\
                    feat_func('POS_N1_N2:%s_%s' % (n1, n2)),\
                    feat_func('POS_P1_N1:%s_%s' % (p1, n1)),\
                    ]
        pos_feat = sorted(feat + filter(lambda x: x != None, pos_feat))
        l1 = len(feat)
        l2 = len(pos_feat)
        f.write('%d %s\n' % (pos, ' '.join('%d:1' % i for i in feat)))
        g.write('%d %s\n' % (pos, ' '.join('%d:1' % i for i in pos_feat)))
        # f.write('%d %s\n' % (pos, ' '.join('%d:%.3f' % (i, 1 / len(feat)) for i in feat)))
        # g.write('%d %s\n' % (pos, ' '.join('%d:%.3f' % (i, 1 / len(pos_feat)) for i in pos_feat)))

    f.close()
    g.close()
    if train:
        mapping.write_stats('feat.txt')
        mapping.save(mapping_file)

def train(instance_file, model_file, param):
    y, x = ll.svm_read_problem(instance_file)
    prob = ll.problem(y, x)
    m = ll.train(prob, param)
    ll.save_model(model_file, m)
    print 'done training', model_file


def predict(input_file, model0_file, model1_file, mapping_file, output_file):
    global bos, eos
    out = open(output_file, 'w')
    m0 = ll.load_model(model0_file)
    m1 = ll.load_model(model1_file)
    mapping = Mapping(mapping_file)
    bos = mapping.map_pos('BOS')
    eos = mapping.map_pos('EOS')

    print '# of features:', len(mapping.feature_dict)

    # for easier mapping from neighbouring pos tags to features
    p_f = {}
    for i in mapping.pos_dict_rev: #[0, 1, 2, 3, ...] bos and eos are also included
        pi = mapping.map_pos_rev(i)
        p_f[(-1, i)] = mapping.map_features('POS_P1:%s' % pi)
        p_f[(-2, i)] = mapping.map_features('POS_P2:%s' % pi)
        p_f[(+1, i)] = mapping.map_features('POS_N1:%s' % pi)
        p_f[(+2, i)] = mapping.map_features('POS_N2:%s' % pi)


        for j in mapping.pos_dict_rev:
            pj = mapping.map_pos_rev(j)
            p_f[(-1, -2, i, j)] = mapping.map_features('POS_P1_P2:%s_%s' % (pi, pj))
            p_f[(+1, +2, i, j)] = mapping.map_features('POS_N1_N2:%s_%s' % (pi, pj))
            p_f[(-1, +1, i, j)] = mapping.map_features('POS_P1_N1:%s_%s' % (pi, pj))

    s1, s2 = 0, 0
    total = 0

    for sent in read_sentence(input_file):
        x_ = []
        g_ = []
        for t in sent:
            feat = t.maxent_features(mapping.map_features)
            x_.append(feat)
            g_.append(mapping.map_pos(t.gold_pos))
        y_1 = map(int, ll.predict([], [{k : 1 for k in f} for f in x_], m0, '-q')[0])

        # y_2 = [choice(xrange(1, len(mapping.pos_dict))) for i in y_1]

        y_2 = inference(m1, p_f, y_1[:], x_)

        for y, y1, y2 in zip(g_, y_1, y_2):
            if y == y1:
                s1 += 1
            if y == y2:
                s2 += 1
            total += 1

        for (t, y) in zip(sent, y_2):
            out.write('%s\t%s\n' % (t.word, mapping.map_pos_rev(y)))
        out.write('\n')

    out.close()
    print 'acc 1: %d / %d = %.4f' % (s1, total, s1 / total)
    print 'acc 2: %d / %d = %.4f' % (s2, total, s2 / total)


# try use logistic regression to sample instead of deterministic way
# instance-wise normalization 
def inference(model, p_f, y_, x_):
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
            feats = pos_feat(p_f, y_, j)
            # cache the prediction of given prev and next tags 
            if feats in cache[j][1]:
                ny = cache[j][1][feats]
            else:
                x = [{k:1 for k in x_[j] + list(feats)}]
                ny = int(ll.predict([], x, model, '-q')[0][0])
                cache[j][1][feats] = ny

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



def pos_feat(p_f, y_, j):
    p1 = y_[j - 1] if j > 0 else bos
    p2 = y_[j - 2] if j > 1 else bos
    n1 = y_[j + 1] if j < len(y_) - 1 else eos
    n2 = y_[j + 2] if j < len(y_) - 2 else eos
    feats = map(p_f.__getitem__, [(-1, p1), (-2, p2), (1, n1), (2, n2), \
                                  (-1, -2, p1, p2), (1, 2, n1, n2), (-1, 1, p1, n1)])
    return tuple(filter(lambda x: x != None, feats))


if __name__ == '__main__':
    if '-a' in sys.argv or '-i' in sys.argv: 
        print 'writing instances...'
        write_instances('../data/pos/train.col', 'train.nopos.inst', 'train.pos.inst', 'maxent.dict', True)
        write_instances('../data/pos/dev.col', 'dev.nopos.inst', 'dev.pos.inst', 'maxent.dict', False)
    if '-a' in sys.argv or '-t' in sys.argv:
        print 'training models...'
        train('train.nopos.inst', 'models/m0.model', '-q -s 5')
        train('train.pos.inst', 'models/m1.model', '-q -s 5')
    if '-a' in sys.argv or '-p' in sys.argv:
        print 'predicting...'
        predict('../data/pos/dev.col', 'models/m0.model', 'models/m1.model', 'maxent.dict', 'dev.predict.col')


