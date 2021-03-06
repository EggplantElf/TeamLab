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
        print f
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
    for i in range(15):
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
    #model.save(model_file)
    print 'done training'
    return best_accuracy, len(model.feature_dict)
    #return model



def predict(filename, model, features_on):
    total = 0
    correct = 0
    output = open('predict_dev.col', 'w')
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
        output.write('\n')
    print 'accuracy: %.4f' % (correct / total)
    output.close()
    return correct / total

def feature_selection():
    t0  = time()
    model_file = 'tmp.dump'
    history = {}
    num_feature_classes = 20
    mask = tuple([True] + [False] * (num_feature_classes-1))
    last_mask = mask
    print "Feature class pattern"
    print ''.join('1' if x else '0' for x in mask)
    accuracy, num_features = train("../data/pos/train.col", "../data/pos/dev.col", model_file, mask)
    history[mask] = (accuracy, num_features)
    print 'accuracy: %.4f\n' % (accuracy)
    
    for i in range(1500):
        success = False
        for j in range(50):
            num = random.randrange(0,num_feature_classes)
            mask = tuple((last_mask[x] if x != num else not last_mask[x]) for x in range(num_feature_classes))
            if mask not in history:
                success = True
                break
        if not success:
            for j in range(50):
                nums = [random.randrange(0,num_feature_classes), random.randrange(0,num_feature_classes)]
                mask = tuple((last_mask[x] if x not in nums else not last_mask[x]) for x in range(num_feature_classes))
                if mask not in history:
                    success = True
                    break
        if not success:
            print "No more possible combinations"
            break                

        print "Feature class pattern"
        print ''.join('1' if x else '0' for x in mask)
        accuracy, num_features = train("../data/pos/train.col", "../data/pos/dev.col", model_file, mask)
        history[mask] = (accuracy, num_features)
        print 'accuracy: %.4f\n' % (accuracy)
        last_accuracy, last_num_features = history[last_mask]
        diff = accuracy - last_accuracy
        if diff > .01 or (diff > 0 and (num_features-last_num_features)/last_num_features < 0.1):
            last_mask = mask
        
    t = time() - t0
    f = open('history_150615.txt', 'w')
    f.write("time used:%d seconds\n" % t)
    for mask, (accuracy, num_features) in sorted(history.items(), key=lambda x:x[1], reverse = True):
        s = ''.join('1' if x else '0' for x in mask)
        f.write('%s\t%.4f\t%d\n' % (s, accuracy, num_features))
    f.close()




if __name__ == '__main__':
    #feature_selection()
    mask = tuple([True] * 20)
    ##model = train("../data/pos/train.col", "../data/pos/dev.col", 'tmp.dump', mask)
    predict("../data/pos/train.col", Model("tmp.dump"), mask)
    
