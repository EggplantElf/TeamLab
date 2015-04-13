from __future__ import division
import sys

def evaluate(gold_file, pred_file):
    gold = []
    pred = []
    for line in open(gold_file):
        if line.strip(): 
            gold.append(line.split()[1].strip())
    for line in open(pred_file):
        if line.strip():
            pred.append(line.split()[1].strip())


    pos = set(gold + pred)
    result = {}
    for p in pos:
        result[p] = [0, 0, 0] # tp, fn, fp

    correct = 0

    for i in range(len(gold)):
        if gold[i] == pred[i]:
            result[gold[i]][0] += 1
            correct += 1
        else:
            result[gold[i]][1] += 1
            result[pred[i]][2] += 1


    print 'Overall accuracy: ', correct / len(gold)

    print 'POS\tPrecision\tRecall\tF-Score'
    for p in result:
        tp, fn, fp = result[p][0], result[p][1], result[p][2]
        if tp + fp == 0:
            precision = 0
        else:
            precision = tp / (tp + fp)
        if tp + fn == 0:
            recall = 0
        else:
            recall = tp / (tp + fn)
        fscore = 2 * tp / (2 * tp + fp + fn)        

        print '%s\t%.2f\t%.2f\t%.2f' % (p, precision, recall, fscore)

if __name__ == '__main__':
    evaluate(sys.argv[1], sys.argv[2])