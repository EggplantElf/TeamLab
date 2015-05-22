import re

class Sentence(list):
    def __init__(self):
        pass

    def __repr__(self):
        return ' '.join(['%s/%s' % (t.word, t.gold_pos) for t in self])


    def add_token(self, line):
        tmp = line.split()
        if len(tmp) > 1:
            word, pos = tmp[0], tmp[1]
        else:
            word, pos = tmp[0], None
        token = Token(self, len(self), word, pos, None)
        self.append(token)


class Token:
    def __init__(self, sent, tid, word, gold_pos = None, pred_pos = '<NA>'):
        self.sent = sent
        self.tid = tid
        self.word = word
        self.gold_pos = gold_pos
        self.pred_pos = pred_pos
        self.get_atom_feats()

    def prev_token(self, offset = 1):
        if self.tid - offset < 0:
            return None
        else:
            return self.sent[self.tid - offset]

    def next_token(self, offset = 1):
        if self.tid + offset >= len(self.sent):
            return None
        else:
            return self.sent[self.tid + offset]

    def prev_pos(self, offset = 1):
        prev_ = self.prev_token(offset)
        if prev_:
            return prev_.gold_pos
        else:
            return 'BOS'

    def next_pos(self, offset = 1):
        next_ = self.next_token(offset)
        if next_:
            return next_.gold_pos
        else:
            return 'EOS'

    def prefix(self, offset):
        if len(self.word) < offset:
            return self.word
        else:
            return self.word[:offset]

    def suffix(self, offset):
        if len(self.word) < offset:
            return self.word
        else:
            return self.word[-offset:]


    def shape(self):
        shapeNC = self.word
        paterns = [['[A-Z]', 'A'], ['[a-z]', 'a'], ['\d',"0"], ['\W',"&"]]

        for i in paterns:
            shapeNC = re.sub(i[0],i[1],shapeNC)
        
        symbols = ['A', 'a', '&', '0']
        shapeC = shapeNC
        for i in symbols:
            shapeC = re.sub(2*i+"+",2*i,shapeC)
        return shapeNC, shapeC


    def get_atom_feats(self):
        self.atom_feats = []
        shapeNC, shapeC = self.shape()
        self.atom_feats.append('WORD:%s' % self.word)
        self.atom_feats.append('SHAPENC:%s' % shapeNC)
        self.atom_feats.append('SHAPEC:%s' % shapeC)
        for i in range(1, 4):
            self.atom_feats.append('PREFIX_%i:%s' % (i, self.prefix(i)))
            self.atom_feats.append('SUFFIX_%i:%s' % (i, self.suffix(i)))

    def maxent_features(self, func):
        features = []
        prev_ = self.prev_token()
        next_ = self.next_token()

        for f in self.atom_feats:
            features.append(func('0~'+f))

        if prev_:
            for f in prev_.atom_feats:
                features.append(func('-1~'+f))
        else:
            features.append(func('BOS'))

        if next_:
            for f in next_.atom_feats:
                features.append(func('+1~'+f))
        else:
            features.append(func('EOS'))

        return sorted(filter(lambda x: x != None, features))


    # def neighbour_pos_features(self, func, prev_pos, next_pos):
    #     features = []
    #     features.append(func('PREV_POS:%s', prev_pos))
    #     features.append(func('NEXT_POS:%s', next_pos))
    #     return sorted(filter(lambda x: x != None, features))






def read_sentence(filename, train = False):
    sentence = Sentence()
    for line in open(filename):
        line = line.strip()
        if line:
            sentence.add_token(line)
        elif len(sentence) != 1:
            yield sentence
            sentence = Sentence()
