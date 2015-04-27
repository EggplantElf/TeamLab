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

    return model


def predict(filename, model):
    total = 0
    correct = 0
    for sent in read_sentence(filename):
        for t in sent:
            f = t.extract_features(model.map_features)
            p = model.predict(f)
            g = model.register_pos(t.gold_pos)
            if p == g:
                correct += 1
            total += 1
    print 'accuracy: %.4f' % (correct / total)




if __name__ == '__main__':
    model = train(sys.argv[1])
    predict(sys.argv[2], model)