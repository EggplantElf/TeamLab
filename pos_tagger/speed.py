import cProfile
import numpy as np
import operator as op
import itertools as it 

def test1():
    w = np.array(xrange(1000000)).reshape(10000, 100)
    f = range(0, 1000, 50)
    for i in xrange(100000):
        a = np.sum(w[i] for i in f) # slower
    print sum(a)

def test2():
    w = np.array(xrange(1000000)).reshape(10000, 100)
    f = range(0, 1000, 50)
    for j in xrange(100000):
        s = np.zeros(100) # , dtype = 'float16' slower
        for i in f:
            s += w[i]
    print sum(s)

    
def test3():
    # w = [[0.0 for i in xrange(100)] for j in xrange(10000)]
    w = np.array(xrange(1000000)).reshape(10000, 100).tolist()
    f = range(0, 1000, 50)
    for j in xrange(100000):
        s = [0.0 for i in xrange(100)]
        for i in f:
            s = it.imap(op.add, s, w[i])
        scores = s

        c, m = None, -9999999
        i = 0
        for s in scores:
            if s > m:
                c = i
                m = s
            i += 1

def test4():
    w = np.array(xrange(1000000)).reshape(10000, 100).tolist()
    f = range(0, 1000, 50)
    for j in xrange(100000):
        s = [0.0 for i in xrange(100)]
        for i in f:
            for j in xrange(100):
                s[j] += w[i][j]
    print sum(s)

def softmax(array):
    e = np.exp(array)
    dist = e / np.sum(e)
    return dist


if __name__ == '__main__':
    cProfile.run('test1()')
    cProfile.run('test2()')
    cProfile.run('test3()')
    # cProfile.run('test4()')