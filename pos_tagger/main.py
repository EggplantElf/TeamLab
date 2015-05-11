from __future__ import division
from sentence import *
from model import *
from random import shuffle
import sys
import cProfile
import itertools as it

def train(train_file, dev_file, model_file):
    model = Model()
    instances = []
    for sent in read_sentence(train_file):
        for t in sent:
            f = t.extract_features(model.register_features)
            instances.append((f, model.register_pos(t.gold_pos)))
    model.create_weights()
    print 'instances:', len(instances)


    dev_instances = []
    for sent in read_sentence(dev_file):
        for t in sent:
            f = t.extract_features(model.map_features)
            dev_instances.append((f, model.register_pos(t.gold_pos)))

    q = 0
    for i in range(20):
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
        
        model.average_for_dev(q)
        correct_dev = 0
        total_dev = 0
        for (f, g) in dev_instances:
            scores = model.get_scores_for_dev(f)
            p = model.predict(scores)
            if p == g:
                correct_dev += 1                
            total_dev += 1

        print 'iteration %d done, train_accuracy: %.4f, dev_accuracy: %.4f' % (i, correct / total, correct_dev / total_dev)

    model.average(q)
    model.save(model_file)
    print 'done training'
    return model



def predict(filename, model):
    total = 0
    correct = 0
    output = open('predict.col', 'w')
    for sent in read_sentence(filename):
        for t in sent:
            f = t.extract_features(model.map_features)
            scores = model.get_scores(f)
            # dist = model.get_dist(scores)
            p = model.predict(scores)
            g = model.register_pos(t.gold_pos)
            # t.pred_pos = model.map_pos_rev(p)
            output.write('%s\t%s\n' % (t.word, model.map_pos_rev(p)))
            if p == g:
                correct += 1
            total += 1
    print 'accuracy: %.4f' % (correct / total)
    output.close()




if __name__ == '__main__':
    model_file = 'tmp.dump'
    model = train(sys.argv[1], sys.argv[2], model_file)
    model = Model(model_file)
    predict(sys.argv[2], model)
