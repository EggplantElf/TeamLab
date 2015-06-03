from __future__ import division
from sentence import *
from mapping import *
from random import shuffle,choice
import sys
sys.path.append('liblinear/python/')
import liblinearutil as ll


def train(instance_file, model_file, param):
    y, x = ll.svm_read_problem(instance_file)
    prob = ll.problem(y, x)
    m = ll.train(prob, param)
    ll.save_model(model_file, m)
    print 'done training', model_file


def predict(instance_file, model_file, param):
    y, x = ll.svm_read_problem(instance_file)
    prob = ll.problem(y, x)
    m = ll.load_model(model_file)
    dist = ll.predict(y, x, m, param)[2]
    print dist[0]


if __name__ == '__main__':
    # train('news20.train.inst', 'tmp.model', '-s 0')
    predict('news20.test.inst', 'tmp.model', '-b 1')