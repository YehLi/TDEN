import os
import sys
import json
import math
import pdb
import time
import pprint
import random
import logging
import argparse
import numpy as np
from datetime import timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import lib.utils as utils
from lib.utils import AverageMeter
from lib.config import cfg, cfg_from_file
from bert.configuration_bert import BertConfig
from lib.tokenization_bert import BertTokenizer
from datasets.data_loader import load_concap_train
from datasets.concept_cap_dataset import ConceptCap
from optimizer.optimizer import Optimizer

class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args

        self.setup_gpu() 
        self.setup_logging()
        self.setup_loader()
        self.setup_network()

    def setup_gpu(self):
        if args.local_rank == -1:
            self.device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
            self.n_gpu = torch.cuda.device_count()
            self.distributed = False
        else:
            torch.cuda.set_device(args.local_rank)
            self.device = torch.device("cuda", args.local_rank)
            self.n_gpu = 1
            torch.distributed.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(minutes=180))
            self.distributed = True
        print("device: {} n_gpu: {}, distributed training: {}".format(
            self.device, self.n_gpu, bool(args.local_rank != -1)))
        if cfg.SEED > 0:
            random.seed(cfg.SEED)
            torch.manual_seed(cfg.SEED)
            torch.cuda.manual_seed_all(cfg.SEED)    

    def setup_logging(self):
        self.logger = logging.getLogger(cfg.LOGGER_NAME)
        self.logger.setLevel(logging.INFO)
        
        if self.distributed and dist.get_rank() > 0:
            return
        
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(levelname)s: %(asctime)s] %(message)s")
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        fh = logging.FileHandler(os.path.join(cfg.ROOT_DIR, cfg.LOGGER_NAME + '.txt'))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        self.logger.info('Training with config:')
        self.logger.info(pprint.pformat(cfg))

    def setup_loader(self):
        self.tokenizer = BertTokenizer.from_pretrained(cfg.TRAIN.BERT_MODEL, do_lower_case=cfg.TRAIN.DO_LOWER_CASE)
        self.train_dataset_loader, self.train_dataset ,self.train_sampler = load_concap_train(args.local_rank, self.tokenizer)

    def setup_network(self):
        config = BertConfig.from_json_file(cfg.CONFIG_FILE)
        if cfg.TRAIN.FROM_PRETRAINED:
            model = BaseBertPreTraining.from_pretrained(cfg.TRAIN.FROM_PRETRAINED, config)
        else:
            model = BaseBertPreTraining(config)
        model.to(self.device)

        if args.local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                model, 
                find_unused_parameters=True,
                device_ids=[self.args.local_rank], 
                output_device=self.args.local_rank,
                broadcast_buffers=False)
        elif self.n_gpu > 1:
            self.model = torch.nn.DataParallel(model)
        else:
            self.model = model

        epoch_steps = len(self.train_dataset_loader)
        n_steps = epoch_steps * cfg.SOLVER.NUM_TRAIN_EPOCHS
        self.optim = Optimizer(self.model, epoch_steps=epoch_steps, n_steps=n_steps)

    def display(self, iteration, batch_time, losses, loss_info):
        if iteration % cfg.SOLVER.DISPLAY != 0:
            return
        if self.distributed and dist.get_rank() > 0:
            return

        info_str = ' (BatchTime: {:.3}) losses = {:.5}'.format(batch_time.avg, losses.avg)
        self.logger.info('Iteration ' + str(iteration) + info_str +', lr = ' +  str(self.optim.get_lr()))
        for name in sorted(loss_info):
            self.logger.info('  ' + name + ' = ' + str(loss_info[name]))

        batch_time.reset()
        losses.reset()

    def snapshot_path(self, name, epoch):
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        return os.path.join(snapshot_folder, name + "_" + str(epoch) + ".bin")

    def save_model(self, epoch):
        if (epoch + 1) % cfg.SOLVER.SNAPSHOT_ITERS != 0:
            return
        if self.distributed and dist.get_rank() > 0:
            return
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        if not os.path.exists(snapshot_folder):
            os.mkdir(snapshot_folder)

        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        torch.save(model_to_save.state_dict(), self.snapshot_path("pytorch_model", epoch+1))

    def train(self):
        max_num_iter = len(self.train_dataset_loader)
        for epochId in range(int(cfg.SOLVER.NUM_TRAIN_EPOCHS)):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epochId)

            self.model.train()
            start = time.time()
            batch_time = AverageMeter()

            for step, batch in enumerate(self.train_dataset_loader):
                iterId = step + (epochId * max_num_iter)

                self.optim.zero_grad()
                batch = tuple(t.cuda(device=self.device, non_blocking=True) for t in batch)
                input_ids, input_mask, segment_ids, lm_label_ids, image_feat, \
                    image_loc, imgfeat_cls_prob, imgfeat_label, imgfeat_mask = (batch)


                # TODO Train model

            self.save_model(epochId)
            if self.distributed:
                dist.barrier()

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='TDEN')
    parser.add_argument('--folder', dest='folder', default=None, type=str)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    if args.folder is not None:
        cfg_from_file(os.path.join(args.folder, 'config.yml'))
    cfg.ROOT_DIR = args.folder

    tokenizer = BertTokenizer.from_pretrained(
        cfg.TRAIN.BERT_MODEL, 
        do_lower_case=cfg.TRAIN.DO_LOWER_CASE
    )
    cfg.MODEL.CLS_ID = tokenizer.vocab["[CLS]"]
    cfg.MODEL.SEP_ID = tokenizer.vocab["[SEP]"]
    cfg.MODEL.MASK_ID = tokenizer.vocab["[MASK]"]

    trainer = Trainer(args)
    trainer.train()