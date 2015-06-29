from __future__ import division


def merge(gold_file, perceptron_file, candis_file, output_file):
    words = []
    y_0 = []
    y_1 = []
    y_2 = []
    y_3 = []
    for line in open(gold_file):
        if line.strip():
            x, y0 = line.strip().split()
            words.append(x)
            y_0.append(y0)
        else:
            words.append('')
            y_0.append('')

    for line in open(perceptron_file):
        if line.strip():
            x, y1 = line.strip().split()
            y_1.append(y1)
        else:
            y_1.append('')

    for line in open(candis_file):
        if line.strip():
            x, y2, y3 = line.strip().split()
            y_2.append(y2)
            y_3.append(y3)
        else:
            y_2.append('')
            y_3.append('')

    # print map(len, [words, y_0, y_1, y_2, y_3])
    s1, s2, s3, s = 0, 0, 0, 0
    total = 0
    o = open(output_file, 'w')
    for i in range(len(y_0)):
        x, y0, y1, y2, y3 = words[i], y_0[i], y_1[i], y_2[i], y_3[i]
        if x:
            if y1 == y2:
                y = y1
            else:
                y = y3
            o.write('%s\t%s\n' % (x, y))

            total += 1
            if y1 == y0:
                s1 += 1
            if y2 == y0:
                s2 += 1
            if y3 == y0:
                s3 += 1
            if y == y0:
                s += 1
        else:
            o.write('\n')
    o.close()

    print 'acc 1: %d / %d = %.4f' % (s1, total, s1 / total)
    print 'acc 2: %d / %d = %.4f' % (s2, total, s2 / total)
    print 'acc 3: %d / %d = %.4f' % (s3, total, s3 / total)
    print 'acc x: %d / %d = %.4f' % (s, total, s / total)


def ehh(gold_file, perceptron_file, candis_file, output_file):
    g = []
    p = []

    for line in open(gold_file):
        g.append(line)

    for line in open(perceptron_file):
        p.append(line)

    for i in range(len(g)):
        if g[i].strip():
            if g[i].split()[0] != p[i].split()[0]:
                print i, g[i]
                exit(0)




if __name__ == '__main__':
    merge('../data/pos/test-nolabels.col', 'test.perceptron.col', 'test.candis.col', 'test.vote.col')
    # ehh('../data/pos/test-nolabels.col', 'test.perceptron.col', 'test.candis.col', 'test.vote.col')
