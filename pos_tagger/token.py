class Token:
    def __init__(self, sent, tid, word, gold_pos = None, pred_pos = None):
        self.sent = sent
        self.tid = tid
        self.word = word
        self.gold_pos = gold_pos
        self.pred_pos = pred_pos

    def prev_token(self, offset = 1):
        if self.tid - offset < 0:
            return '<BOS>'
        else:
            return self.sent[self.tid - offset]

    def next_token(self, offset = 1):
        if self.tid + offset >= len(sent):
            return '<EOS>'
        else:
            return self.sent[self.tid + offset]

    def extract_features(self, func):
        features = []
        features.append(func('WORD:%s' % self.word))
        for i in range(1, 4):
            features.append(func('PREV_WORD_%i:%s' % (i, self.prev_token())))
        for i in range(1, 4):
            features.append(func('NEXT_WORD_%i:%s' % (i, self.next_token())))
