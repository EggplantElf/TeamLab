from __future__ import division
from sentence import *
from model import *
from random import shuffle
import sys
import cProfile
import itertools as it

def train(train_file, model_file):
    model = Model()
    instances = []
    for sent in read_sentence(train_file):
        for t in sent:
            f = t.extract_features(model.register_features)
            instances.append((f, model.register_pos(t.gold_pos)))
    model.create_weights()
    print 'instances:', len(instances)

    q = 0
    for i in range(10):
        total = 0
        correct = 0
        # shuffle(instances)
        for (f, g) in instances:
            scores = model.get_scores(f)
            p = model.predict(scores)
            if p != g:
                # model.update(f, g, p)
                model.update_pa(f, g, p, scores, q, 0.1)
            else:
                correct += 1
            total += 1
            q += 1
        print 'iteration %d done, accuracy: %.4f' % (i, correct / total)

    model.average(q)
    save(model, model_file)
    print 'done training'
    # return model



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
    model_file = 'tmp.dump'
    train(sys.argv[1], model_file)
    model = load(model_file)
    predict(sys.argv[2], model)
