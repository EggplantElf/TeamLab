from __future__ import division
from sentence import *
from model import *
import sys

def train(filename):
    model = Model()
    instances = []
    for sent in read_sentence(filename):
        for t in sent:
            f = t.extract_features(model.register_features)
            instances.append((f, model.register_pos(t.gold_pos)))
    model.create_weights()

    for i in range(10):
        total = 0
        correct = 0
        for (f, g) in instances:
            p = model.predict(f)
            if p != g:
                model.update(f, g, p)
            else:
                correct += 1
            total += 1
        print 'iteration %d done, accuracy: %.4f' % (i, correct / total)





if __name__ == '__main__':
    train(sys.argv[1])