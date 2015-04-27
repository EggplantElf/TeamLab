class Model:
    def __init(self):
        self.feat_dict = {}

    def map(self, feat_str):
        if feat_str in feat_dict:
            return feat_dict[feat_str]
        else:
            return -1

    def register(self, feat_str):
        if feat_str not in feat_str:
            feat_dict[feat_str] = len(feat_str)
        return feat_dict[feat_str]


