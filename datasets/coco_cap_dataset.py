import os
import json
import _pickle as cPickle
import logging
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import lib.utils as utils
from lib.config import cfg

def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)

def load_dataset(dataroot, name):
    imglist_path = os.path.join(dataroot, "coco_%s_image_id.txt" % name)
    caption_path = os.path.join(dataroot, 'dataset_coco.json')
    with open(imglist_path) as fid:
        imglist = [int(line.strip()) for line in fid]
    imglist = set(imglist)
    
    entries = []
    annotation = json.load(open(caption_path, 'r'))['images']
    for img in annotation:
        image_id = img['cocoid']
        if image_id not in imglist:
            continue

        sentences = []
        for sent in img['sentences']:
           sentences.append(sent['raw'].lower().strip().strip('.'))
        entries.append({"caption": sentences, "image_id": image_id})

    return entries

class COCOCapDataset(Dataset):
    def __init__(
        self,
        task_name,
        dataroot,
        anno_path,
        split,
        feat_folder,
        gt_feat_folder,
        tokenizer,
        padding_index=0,
        max_seq_length=16,
        max_region_num=51
    ):
        self.split = split
        self.max_region_num = max_region_num
        self.max_seq_length = max_seq_length
        self.feat_folder = feat_folder
        self.tokenizer = tokenizer
        self.padding_index = padding_index
        self.num_labels = max_seq_length
        self.mask_token_id = cfg.MODEL.MASK_ID
        self.sep_token_id = cfg.MODEL.SEP_ID
        self.seq_per_img = 5

        cache_path = os.path.join(
            dataroot,
            "cache",
            task_name + "_" + split + "_" + str(max_seq_length) + ".pkl",
        )

        if not os.path.exists(cache_path):
            self.entries = load_dataset(dataroot, split)
            self.tokenize(max_seq_length)
            cPickle.dump(self.entries, open(cache_path, "wb"))
        else:
            self.entries = cPickle.load(open(cache_path, "rb"))

    def tokenize(self, max_length=16):
        for entry in self.entries:
            entry['input_seq'] = []
            entry['target_seq'] = []
            entry['segment_ids'] = []

            for sent in entry['caption']:
                tokens = self.tokenizer.encode(sent)
                input_seq = [cfg.MODEL.CLS_ID] + tokens
                target_seq = tokens + [cfg.MODEL.SEP_ID]

                input_seq = input_seq[: max_length]
                target_seq = target_seq[: max_length]
                segment_ids = [1] * max_length

                if len(input_seq) < max_length:
                    padding = [self.padding_index] * (max_length - len(input_seq))
                    tpadding = [-1] * (max_length - len(input_seq))
                    input_seq = input_seq + padding
                    target_seq = target_seq + tpadding

                assert_eq(len(input_seq), max_length)
                assert_eq(len(target_seq), max_length)

                entry['input_seq'].append(input_seq)
                entry['target_seq'].append(target_seq)
                entry['segment_ids'].append(segment_ids)
            entry.pop('caption')

    def __getitem__(self, index):
        entry = self.entries[index]
        image_id = entry["image_id"]
        
        feat_folder = self.feat_folder
        features, num_boxes, boxes = utils.image_features_reader(feat_folder, image_id)

        mix_num_boxes = min(int(num_boxes), self.max_region_num)
        mix_boxes_pad = np.zeros((self.max_region_num, 5))
        mix_features_pad = np.zeros((self.max_region_num, 2048))

        image_mask = [1] * (int(mix_num_boxes))
        while len(image_mask) < self.max_region_num:
            image_mask.append(0)

        mix_boxes_pad[:mix_num_boxes] = boxes[:mix_num_boxes]
        mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]

        features = torch.tensor(mix_features_pad).float()
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(mix_boxes_pad).float()

        input_seq = np.zeros((self.seq_per_img, self.max_seq_length), dtype='int')
        target_seq = np.zeros((self.seq_per_img, self.max_seq_length), dtype='int')
        segment_ids = np.zeros((self.seq_per_img, self.max_seq_length), dtype='int')

        if self.split == 'train':
            sents_num = len(entry['input_seq'])
            if sents_num >= self.seq_per_img:
                sid = 0
                ixs = random.sample(range(sents_num), self.seq_per_img)
            else:
                sid = sents_num
                ixs = random.sample(range(sents_num), self.seq_per_img - sents_num)
                input_seq[0:sents_num, :] = entry['input_seq']
                target_seq[0:sents_num, :] = entry['target_seq']
                segment_ids[0:sents_num, :] = entry['segment_ids']

            for i, ix in enumerate(ixs):
                input_seq[sid + i] = entry['input_seq'][ix]
                target_seq[sid + i] = entry['target_seq'][ix]
                segment_ids[sid + i] = entry['segment_ids'][ix]

            input_mask = torch.tril(torch.ones(
                (self.max_seq_length, self.max_seq_length), dtype=torch.long))
            input_mask = input_mask.unsqueeze(0).expand([self.seq_per_img, self.max_seq_length, self.max_seq_length])
            
        else:
            input_seq = np.array([cfg.MODEL.CLS_ID] * self.max_seq_length)
            target_seq = np.array([-1] * self.max_seq_length)
            segment_ids = np.array([1] * self.max_seq_length)
            input_mask = torch.tril(torch.ones(
                (self.max_seq_length, self.max_seq_length), dtype=torch.long))
         
        return (
            features,
            spatials,
            image_mask,
            input_seq,
            target_seq,
            input_mask,
            segment_ids,
            image_id,
        )

    def __len__(self):
        return len(self.entries)

    def decode_sequence(self, seq):
        N, T = seq.size()
        seq = seq.data.cpu().numpy()
        sents = []
        for n in range(N):
            words = []
            for t in range(T):
                ix = seq[n, t]
                if ix == self.sep_token_id:
                    break
                words.append(self.tokenizer.ids_to_tokens[ix])
            sent = self.tokenizer.convert_tokens_to_string(words)
            sents.append(sent)
        return sents

class COCOTestCapDataset(Dataset):
    def __init__(
        self,
        task_name,
        dataroot,
        anno_path,
        split,
        feat_folder,
        gt_feat_folder,
        tokenizer,
        padding_index=0,
        max_seq_length=16,
        max_region_num=51
    ):
        self.split = split
        self.max_region_num = max_region_num
        self.max_seq_length = max_seq_length
        self.feat_folder = feat_folder
        self.tokenizer = tokenizer
        self.padding_index = padding_index
        self.num_labels = max_seq_length
        self.mask_token_id = cfg.MODEL.MASK_ID
        self.sep_token_id = cfg.MODEL.SEP_ID
        self.seq_per_img = 5

        imglist_path = os.path.join(dataroot, "coco_test4w_image_id.txt")
        with open(imglist_path) as fid:
            self.image_id = [int(line.strip()) for line in fid]

    def __len__(self):
        return len(self.image_id)

    def decode_sequence(self, seq):
        N, T = seq.size()
        seq = seq.data.cpu().numpy()
        sents = []
        for n in range(N):
            words = []
            for t in range(T):
                ix = seq[n, t]
                if ix == self.sep_token_id:
                    break
                words.append(self.tokenizer.ids_to_tokens[ix])
            sent = self.tokenizer.convert_tokens_to_string(words)
            sents.append(sent)
        return sents

    def __getitem__(self, index):
        image_id = self.image_id[index]
        
        feat_folder = self.feat_folder
        features, num_boxes, boxes = utils.image_features_reader(feat_folder, image_id)

        mix_num_boxes = min(int(num_boxes), self.max_region_num)
        mix_boxes_pad = np.zeros((self.max_region_num, 5))
        mix_features_pad = np.zeros((self.max_region_num, 2048))

        image_mask = [1] * (int(mix_num_boxes))
        while len(image_mask) < self.max_region_num:
            image_mask.append(0)

        mix_boxes_pad[:mix_num_boxes] = boxes[:mix_num_boxes]
        mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]

        features = torch.tensor(mix_features_pad).float()
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(mix_boxes_pad).float()

        input_seq = np.array([cfg.MODEL.CLS_ID] * self.max_seq_length)
        target_seq = np.array([-1] * self.max_seq_length)
        segment_ids = np.array([1] * self.max_seq_length)
        input_mask = torch.tril(torch.ones(
            (self.max_seq_length, self.max_seq_length), dtype=torch.long))
        
        return (
            features,
            spatials,
            image_mask,
            input_seq,
            target_seq,
            input_mask,
            segment_ids,
            image_id,
        )
