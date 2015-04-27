import numpy as np


class Model:
    def __init__(self):
        self.feature_dict = {'#': 0}
        self.pos_dict = {}
        self.pos_dict_rev = {}

    def map_features(self, feature):
        if feature in self.feature_dict:
            return self.feature_dict[feature]
        else:
            return -1

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

    def map_pos_rev(self, index):
        return self.pos_dict_rev[index]

    def create_weights(self):
        self.weights = np.zeros(shape = (len(self.feature_dict), len(self.pos_dict)))


    def get_scores(self, features):
        s = np.sum(self.weights[i] for i in features)
        return s

    def predict(self, features):
        scores = self.get_scores(features)
        m = max(enumerate(scores), key = (lambda x: x[1])) # e.g. (4, 10.7) meaning the max score is the 4th pos with score of 10.7
        return m[0]

    def update(self, f, g, p):
        for i in f:
            self.weights[i, g] += 1
            self.weights[i, p] -= 1





