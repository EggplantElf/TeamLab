from __future__ import division
from sentence import *
from model import *
from random import shuffle
import sys
import cProfile
import itertools as it
import random
from time import time

def train(train_file, dev_file, model_file, features_on):
    model = Model()
    instances = []
    for sent in read_sentence(train_file):
        for t in sent:
            f = t.extract_features(model.register_features, features_on)
            instances.append((f, model.register_pos(t.gold_pos)))
    model.create_weights()
    print 'instances:', len(instances)


    dev_instances = []
    for sent in read_sentence(dev_file):
        for t in sent:
            f = t.extract_features(model.map_features, features_on)
            dev_instances.append((f, model.register_pos(t.gold_pos)))

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

        print 'iteration %d done, train_accuracy: %.4f, dev_accuracy: %.4f' % (i+1, correct / total, correct_dev / total_dev)
        if correct_dev / total_dev > best_accuracy:
            best_accuracy = correct_dev / total_dev
    model.average(q)
    print 'done training'
    return best_accuracy



def predict(filename, model, features_on):
    total = 0
    correct = 0
    output = open('predict.col', 'w')
    for sent in read_sentence(filename):
        for t in sent:
            f = t.extract_features(model.map_features, features_on)
            scores = model.get_scores(f)
            # dist = model.get_dist(scores)
            p = model.predict(scores)
            g = model.register_pos(t.gold_pos)
            # t.pred_pos = model.map_pos_rev(p)
            output.write('%s\t%s\n' % (t.word, model.map_pos_rev(p)))
            if p == g:
                correct += 1
            total += 1
    #print 'accuracy: %.4f' % (correct / total)
    output.close()
    return correct / total



if __name__ == '__main__':
    t0  = time()
    model_file = 'tmp.dump'
    history = {}
    num_features = 20
    mask = tuple([True] * num_features)
    last_mask = mask
    print "Feature class pattern"
    print ''.join('1' if x else '0' for x in mask)
    accuracy = train(sys.argv[1], sys.argv[2], model_file, mask)
    history[mask] = accuracy
    print 'accuracy: %.4f\n' % (accuracy)
    
    for i in range(4):
        success = False
        for j in range(50):
            num = random.randrange(0,num_features)
            mask = tuple((last_mask[x] if x != num else not last_mask[x]) for x in range(num_features))
            if mask not in history:
                success = True
                break
        if not success:
            for j in range(50):
                nums = [random.randrange(0,num_features), random.randrange(0,num_features)]
                mask = tuple((last_mask[x] if x not in nums else not last_mask[x]) for x in range(num_features))
                if mask not in history:
                    success = True
                    break
        if not success:
            print "No more possible combinations"
            break                

        print "Feature class pattern"
        print ''.join('1' if x else '0' for x in mask)
        accuracy = train(sys.argv[1], sys.argv[2], model_file, mask)
        history[mask] = accuracy
        print 'accuracy: %.4f\n' % (accuracy)
        if history[last_mask] < accuracy:
            last_mask = mask
        else:
            p = random.random()
            if p < 0.1 and history[last_mask] - accuracy < 0.0005:
                last_mask = mask
    t = time() - t0
    f = open('history.txt', 'w')
    f.write("time used:%d seconds" % t)
    for mask, accuracy in sorted(history.items(), key=lambda x:x[1], reverse = True):
        s = ''.join('1' if x else '0' for x in mask)
        f.write('%s\t%.4f\n' % (s, accuracy))
    f.close()