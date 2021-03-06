import cPickle
import gzip

class Mapping:
    def __init__(self, mapping_file = None):
        if mapping_file:
            self.load(mapping_file)
        else:
            self.feature_dict = {'#': 0}
            self.pos_dict = {'#': 0}
            self.pos_dict_rev = {0: '#'}
            self.stats = {'#': 0}

    def save(self, mapping_file):
        stream = gzip.open(mapping_file,'wb')
        cPickle.dump(self.feature_dict, stream, -1)
        cPickle.dump(self.pos_dict, stream, -1)
        cPickle.dump(self.pos_dict_rev, stream, -1)
        stream.close()

    def load(self, mapping_file):
        stream = gzip.open(mapping_file,'rb')
        self.feature_dict = cPickle.load(stream)
        self.pos_dict = cPickle.load(stream)
        self.pos_dict_rev = cPickle.load(stream)
        stream.close()

    def write_stats(self, feat_file):
        o = open(feat_file, 'w')
        for k in sorted(self.stats):
            l = len(k)
            o.write('%s%s%d  \n' % (k, ' '* (30 -l), self.stats[k]))
        o.close()


    def register_features(self, feature):
        if feature not in self.feature_dict:
            self.feature_dict[feature] = len(self.feature_dict)
            self.stats[feature] = 0
        self.stats[feature] += 1
        return self.feature_dict[feature]

    def register_pos(self, pos):
        if pos not in self.pos_dict:
            i = len(self.pos_dict)
            self.pos_dict[pos] = i
            self.pos_dict_rev[i] = pos
        return self.pos_dict[pos]

    def map_features(self, feature):
        return self.feature_dict.get(feature, None)

    def map_pos(self, pos):
        return self.pos_dict.get(pos, None)

    def map_pos_rev(self, index):
        return self.pos_dict_rev.get(index, None)

