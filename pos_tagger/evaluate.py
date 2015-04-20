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
    
    if len(pred) != len(gold):
        print "The gold file and the prediction file don't correspond"
        exit(0)
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

    print 'POS\tPrecision\tRecall\tF-Score'
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

        print '%s\t%.2f\t%.2f\t%.2f' % (p, precision, recall, fscore)
        
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
