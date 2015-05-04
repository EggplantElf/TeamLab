from __future__ import division
from sentence import *
from model import *
from random import shuffle
import sys
import cProfile
import itertools as it

def train(filename):
    model = Model()
    instances = []
    for sent in read_sentence(filename):
        for t in sent:
            f = t.extract_features(model.register_features)
            instances.append((f, model.register_pos(t.gold_pos)))
    model.create_weights()
    print 'instances:', len(instances)

    q = 0
    for i in range(15):
        total = 0
        correct = 0
        # shuffle(instances)
        for (f, g) in instances:
            scores = model.get_scores(f)
            p = model.predict(scores)
            if p != g:
                model.update(f, g, p, scores, q, 0.5)
            else:
                correct += 1
            total += 1
            q += 1
        print 'iteration %d done, accuracy: %.4f' % (i, correct / total)

    # model.average(q)
    return model



def predict(filename, model):
    total = 0
    correct = 0
    for sent in read_sentence(filename):
        for t in sent:
            f = t.extract_features(model.map_features)
            scores = model.get_scores(f)
            # dist = model.get_dist(scores)
            p = model.predict(scores)
            g = model.register_pos(t.gold_pos)
            t.pred_pos = model.map_pos_rev(p)
            if p == g:
                correct += 1
            total += 1
    print 'accuracy: %.4f' % (correct / total)




if __name__ == '__main__':
    model = train(sys.argv[1])
    predict(sys.argv[2], model)
    # cProfile.run('model = train(sys.argv[1])')
    # cProfile.run('predict(sys.argv[2], model)')
