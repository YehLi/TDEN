import os
import sys
import numpy as np
import pickle
from lib.config import cfg

from scorer.cider import Cider
from lib.tokenization_bert import BertTokenizer

factory = {
    'Cider': Cider,
}

class Scorer(object):
    def __init__(self):
        super(Scorer, self).__init__()
        self.scorers = []
        self.weights = cfg.SCORER.WEIGHTS
        self.gts = pickle.load(open(cfg.SCORER.GT_PATH, 'rb'), encoding='bytes')
        for name in cfg.SCORER.TYPES:
            self.scorers.append(factory[name]())

        self.tokenizer = BertTokenizer.from_pretrained(cfg.TRAIN.BERT_MODEL, do_lower_case=cfg.TRAIN.DO_LOWER_CASE)

    def get_sents(self, sent):
        words = []
        for word in sent:
            if word == cfg.MODEL.SEP_ID:
                words.append('.')
                break
            words.append(self.tokenizer.ids_to_tokens[word])

        words = self.tokenizer.convert_tokens_to_string(words).split()
        return words

    def __call__(self, ids, res):
        hypo = [self.get_sents(r) for r in res]
        gts = [self.gts[i] for i in ids]

        rewards_info = {}
        rewards = np.zeros(len(ids))
        for i, scorer in enumerate(self.scorers):
            score, scores = scorer.compute_score(gts, hypo)
            rewards += self.weights[i] * scores
            rewards_info[cfg.SCORER.TYPES[i]] = score
        return rewards, rewards_info