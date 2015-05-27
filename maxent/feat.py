import gzip
import cPickle

def feat(dict_file, output_file):
    stream = gzip.open(dict_file,'rb')
    feature_dict = cPickle.load(stream)
    o = open(output_file, 'w')
    for k in sorted(feature_dict):
        l = len(k)
        o.write('%s%s%d  \n' % (k, ' '* (30 -l), feature_dict[k]))
    o.close()

if __name__ == '__main__':
    feat('maxent.dict', 'feat.txt')