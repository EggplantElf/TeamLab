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
    def __init__(self, sent, tid, word, gold_pos = None, pred_pos = None):
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
        return ''




    def extract_features(self, func):
        # initially with x_0, indices of the features starts from 1
        features = [0] 

        # word features
        features.append(func('WORD:%s' % self.word))
        # features.append(func('SHAPE:%s' % self.shape()))

        for i in range(1, 4):
            features.append(func('PREFIX_%i:%s' % (i, self.prefix(i))))
            features.append(func('SUFFIX_%i:%s' % (i, self.suffix(i))))


        # context features
        for i in range(1, 4):
            features.append(func('PREV_WORD_%i:%s' % (i, self.prev_word(i))))
        for i in range(1, 4):
            features.append(func('NEXT_WORD_%i:%s' % (i, self.next_word(i))))
        for i in range(1, 4):
            features.append(func('PREV_POS_%i:%s' % (i, self.prev_pos(i))))

        return sorted(features)


def read_sentence(filename, train = False):
    sentence = Sentence()
    for line in open(filename):
        line = line.strip()
        if line:
            sentence.add_token(line)
        elif len(sentence) != 1:
            yield sentence
            sentence = Sentence()

