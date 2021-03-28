import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg

def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = one_hots * labels
    return scores

def rl_criterion(seq, logP, rewards):
    mask = (seq > 0) & (seq != cfg.MODEL.SEP_ID)
    mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)
    rewards = rewards.view(-1, 1).expand_as(logP)
    logP = torch.masked_select(logP, mask)
    rewards = torch.masked_select(rewards, mask)
    loss = torch.mean(-logP * rewards)
    return loss
