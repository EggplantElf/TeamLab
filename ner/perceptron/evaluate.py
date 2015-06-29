from __future__ import division
import sys


def evaluate(gold_file, pred_file):
    gold = []
    pred = []
    # read the gold and prediction files
    for line in open(gold_file):
        if line.strip(): 
            gold.append(line.split()[1].strip())
    for line in open(pred_file):
       if line.strip():
           pred.append(line.split()[1].strip())
    
    # make sure the two files contains same amount of data
    if len(pred) != len(gold):
        print "The gold file and the prediction file don't correspond"
        exit(0)

    pos = set(gold + pred)
    result = {}

    # initialize the counters for true positive, false negative, false positive for each pos tag
    for p in pos:
        result[p] = [0, 0, 0] # tp, fn, fp
    correct = 0

    # store the tp, fn, fp 
    for i in range(len(gold)):
        if gold[i] == pred[i]:
            result[gold[i]][0] += 1
            correct += 1
        else:
            result[gold[i]][1] += 1
            result[pred[i]][2] += 1

    # calculate the precision, recall and f-score for each pos tag
    print 'POS\tPrecision  Recall  F-Score'
    for p in result:
        tp, fn, fp = result[p][0], result[p][1], result[p][2]
        if tp + fp == 0:
            precision = 0
        else:
            precision = tp / (tp + fp)
        result[p].append(precision)
        if tp + fn == 0:
            recall = 0
        else:
            recall = tp / (tp + fn)
        result[p].append(recall)
        fscore = 2 * tp / (2 * tp + fp + fn) 
        result[p].append(fscore)       

        print '%s\t  %.2f\t    %.2f    %.2f' % (p, precision, recall, fscore)
        
    # calculate the macro average precision, recall and f-score 
    print "Average Precision:", sum(result[p][3] for p in result)/len(result)
    print "Average Recall:", sum(result[p][4] for p in result)/len(result)
    print "Average F-Score:", sum(result[p][5] for p in result)/len(result)
    print 'Overall accuracy: ', correct / len(gold)
    
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "The number of name files is not correct"
        exit(0)
    else:
        evaluate(sys.argv[1], sys.argv[2])
