import os
import numpy as np
import torch
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler

from lib.config import cfg
from datasets.concept_cap_dataset import ConceptCap


def load_concap_train(local_rank, tokenizer):
    concept_cap = ConceptCap(
        task_name = 'concap',
        dataroot = cfg.PRETRAIN.DATAROOT,
        anno_file = cfg.PRETRAIN.ANNO,
        feat_folder = cfg.DATA_LOADER.PRETRAIN_FEAT_FOLDER,
        tokenizer = tokenizer,
        bert_model = cfg.TRAIN.BERT_MODEL,
        padding_index=0,
        max_seq_length=cfg.PRETRAIN.MAX_SEQ_LEN,
        max_region_num=cfg.PRETRAIN.MAX_REGION_NUM,
    )

    if local_rank == -1:
        train_sampler = RandomSampler(concept_cap)
        sampler = None
    else:
        train_sampler = DistributedSampler(concept_cap)
        sampler = train_sampler

    loader = torch.utils.data.DataLoader(
        concept_cap, 
        batch_size = cfg.PRETRAIN.BATCH_SIZE,
        num_workers = cfg.DATA_LOADER.NUM_WORKERS, 
        drop_last = cfg.DATA_LOADER.DROP_LAST, 
        pin_memory = cfg.DATA_LOADER.PIN_MEMORY,
        sampler = train_sampler)
    return loader, concept_cap, sampler