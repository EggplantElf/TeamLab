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


    def prev_word(self, offset = 1):
        if self.tid - offset < 0:
            return '<BOS>'
        else:
            return self.sent[self.tid - offset].word

    def next_word(self, offset = 1):
        if self.tid + offset >= len(self.sent):
            return '<EOS>'
        else:
            return self.sent[self.tid + offset].word


    def prev_pos(self, offset = 1):
        if self.tid - offset < 0:
            return '<BOS>'
        else:
            return self.sent[self.tid - offset].pred_pos

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




    def extract_features(self, func, features_on):
        # initially with x_0, indices of the features starts from 1
        features = [0] 

        # word features
        if features_on[0]:
            features.append(func('WORD:%s' % self.word))
        shapeNC, shapeC = self.shape()

        if features_on[1]:
            features.append(func('SHAPENC:%s' % shapeNC))

        if features_on[2]:
            features.append(func('SHAPEC:%s' % shapeC))

        for i in range(1, 4):
            if features_on[2+i]:
                features.append(func('PREFIX_%i:%s' % (i, self.prefix(i))))
            if features_on[5+i]:
                features.append(func('SUFFIX_%i:%s' % (i, self.suffix(i))))


        # context features
        for i in range(1, 4):
            if features_on[8+i]:
                features.append(func('PREV_WORD_%i:%s' % (i, self.prev_word(i))))
        for i in range(1, 4):
            if features_on[11+i]:
                features.append(func('NEXT_WORD_%i:%s' % (i, self.next_word(i))))

        # bigram features
        if features_on[15]:
            features.append(func('PREV+THIS_WORD:%s+%s' % (self.prev_word(1), self.word)))
        if features_on[16]:
            features.append(func('NEXT+THIS_WORD:%s+%s' % (self.next_word(1), self.word)))
        
        # trigram features
        if features_on[17]:
            features.append(func('PREV+PREV+THIS_WORD:%s+%s+%s' % (self.prev_word(2), self.prev_word(1), self.word)))
        if features_on[18]:
	    features.append(func('NEXT+NEXT+THIS_WORD:%s+%s+%s' % (self.next_word(2), self.next_word(1), self.word)))
        if features_on[19]:
            features.append(func('NEXT+NEXT+THIS_WORD:%s+%s+%s' % (self.prev_word(1), self.word, self.next_word(1))))

        # don't register here, add while training
        # for i in range(1, 3):
        #     features.append(func('PREV_POS_%i:%s' % (i, self.prev_pos(i))))

        return sorted(filter(lambda x: x != None, features))


def read_sentence(filename, train = False):
    sentence = Sentence()
    for line in open(filename):
        line = line.strip()
        if line:
            sentence.add_token(line)
        elif len(sentence) != 1:
            yield sentence
            sentence = Sentence()

