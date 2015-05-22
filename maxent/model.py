import cPickle
import gzip

class Model:
    def __init__(self, modelfile = None):
        if modelfile:
            self.load(modelfile)
        else:
            self.feature_dict = {'#': 0}
            self.pos_dict = {'#': 0}
            self.pos_dict_rev = {0: '#'}

    def save(self, modelfile):
        stream = gzip.open(modelfile,'wb')
        cPickle.dump(self.feature_dict, stream, -1)
        cPickle.dump(self.pos_dict, stream, -1)
        cPickle.dump(self.pos_dict_rev, stream, -1)
        stream.close()

    def load(self, modelfile):
        stream = gzip.open(modelfile,'rb')
        self.feature_dict = cPickle.load(stream)
        self.pos_dict = cPickle.load(stream)
        self.pos_dict_rev = cPickle.load(stream)
        stream.close()


    def register_features(self, feature):
        if feature not in self.feature_dict:
            self.feature_dict[feature] = len(self.feature_dict)
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

