from __future__ import division
import sys
from collections import defaultdict


# assume already that there's no case like "B-PER, I-LOC", also not start with "I-LOC" 
def evaluate_ner(gold_file, pred_file):
    gold, pred = set(), set()

    j = 0
    b = 0
    e = 'X'
    for line in open(gold_file):
        line = line.strip()
        j += 1
        # print j, line
        if line:
            tag = line.split('|')[1]
            if tag == 'O':
                if b:
                    gold.add((b, j, e)) # (1, 3, 'PER') for B-PER, I-PER, O
                    b, e = 0, 'X'
            else:
                c, t  = tag.split('-') # c = 'B', t = 'PER'
                if c == 'B':
                    if b: # B-PER, B-LOC, which is allowed
                        gold.add((b, j, e))
                    e, b = t, j # e = 'PER', b = 4
        else: # EOS
            if b: # sentence ends not with 'O'
                gold.add((b, j, e)) # (1, 3, 'PER') for B-PER, I-PER, O
                b, e = 0, 'X'

    j = 0
    b = 0
    e = 'X'
    for line in open(pred_file):
        line = line.strip()
        j += 1
        # print j, line
        if line:
            tag = line.split('|')[1]
            if tag == 'O':
                if b:
                    pred.add((b, j, e)) # (1, 3, 'PER') for B-PER, I-PER, O
                    b, e = 0, 'X'
            else:
                c, t  = tag.split('-') # c = 'B', t = 'PER'
                if c == 'B':
                    if b: # B-PER, B-LOC, which is allowed
                        pred.add((b, j, e))
                    e, b = t, j # e = 'PER', b = 4
        else: # EOS
            if b: # sentence ends not with 'O'
                pred.add((b, j, e)) # (1, 3, 'PER') for B-PER, I-PER, O
                b, e = 0, 'X'

    result = {}
    # initialize the counters for true positive, false negative, false positive for each pos tag
    for p in ['PER', 'LOC', 'ORG', 'MISC']:
        result[p] = [0, 0, 0] # tp, fn, fp
    correct = 0

    # store the tp, fn, fp 
    same = gold & pred
    no_pred = gold - pred
    no_gold = pred - gold

    for s, e, t in gold & pred:
        result[t][0] += 1
        correct += 1

    for s, e, t in gold - pred:
        result[t][1] += 1

    for s, e, t in pred - gold:
        result[t][2] += 1


    # calculate the precision, recall and f-score for each pos tag
    print 'NER\tPrecision  Recall  F-Score'
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
        evaluate_ner(sys.argv[1], sys.argv[2])
