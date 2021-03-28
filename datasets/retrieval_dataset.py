import json
from typing import Any, Dict, List
import random
import os

import torch
from torch.utils.data import Dataset
import numpy as np
import _pickle as cPickle

import jsonlines
import sys
import pdb
import lib.utils as utils

def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)

def load_annotations(annotations_jsonpath):
    with jsonlines.open(annotations_jsonpath) as reader:
        entries = []
        imgid2entry = {}
        count = 0

        for annotation in reader:
            image_id = int(annotation["img_path"].split(".")[0])
            imgid2entry[image_id] = []
            for sentences in annotation["sentences"]:
                entries.append({"caption": sentences, "image_id": image_id})
                imgid2entry[image_id].append(count)
                count += 1

    return entries, imgid2entry

class RetrievalDataset(Dataset):
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
        max_seq_length=20,
        max_region_num=51
    ):
        self.entries, self.imgid2entry = load_annotations(anno_path)
        self.image_id_list = [*self.imgid2entry]

        self.feat_folder = feat_folder
        self.tokenizer = tokenizer
        self.num_labels = 1
        self.split = split
        self.padding_index = padding_index
        self.max_region_num = max_region_num
        self.max_seq_length = max_seq_length

        cache_path = os.path.join(
            dataroot,
            "cache",
            task_name + "_" + split + "_" + str(max_seq_length) + ".pkl",
        )

        if not os.path.exists(cache_path):
            self.tokenize()
            cPickle.dump(self.entries, open(cache_path, "wb"))
        else:
            data = cPickle.load(open(cache_path, "rb"))
            self.entries = data['entries']
            self.imgid2entry = data['imgid2entry']

    def tokenize(self):
        for entry in self.entries:
            tokens = self.tokenizer.encode(entry["caption"])
            tokens = tokens[: self.max_seq_length - 2]
            tokens = self.tokenizer.add_special_tokens_single_sentence(tokens)

            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)

            if len(tokens) < self.max_seq_length:
                # Note here we pad in front of the sentence
                padding = [self.padding_index] * (self.max_seq_length - len(tokens))
                tokens = tokens + padding
                input_mask += padding
                segment_ids += padding

            assert_eq(len(tokens), self.max_seq_length)
            entry["token"] = tokens
            entry["input_mask"] = input_mask
            entry["segment_ids"] = segment_ids
            entry.pop("caption")

    def load_feat_sent(self, entry):
        image_id = entry["image_id"]
        features, num_boxes, boxes = utils.image_features_reader(self.feat_folder, image_id)

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

        caption = torch.from_numpy(np.array(entry["token"]))
        input_mask = torch.from_numpy(np.array(entry["input_mask"]))
        segment_ids = torch.from_numpy(np.array(entry["segment_ids"]))

        return features, image_mask, spatials, caption, input_mask, segment_ids

    def __getitem__(self, index):
        entry = self.entries[index]
        image_id = entry["image_id"]

        features1, image_mask1, spatials1, caption1, input_mask1, segment_ids1 = self.load_feat_sent(entry)
        
        # similar to vil-bert
        while True:
            # sample a random image:
            img_id2 = random.choice(self.image_id_list)
            if img_id2 != image_id:
                break

        entry2 = self.entries[random.choice(self.imgid2entry[img_id2])]
        image_id2 = entry2["image_id"]
        features2, image_mask2, spatials2, caption2, input_mask2, segment_ids2 = self.load_feat_sent(entry2)
        
        features = torch.stack([features1, features2], dim=0)
        spatials = torch.stack([spatials1, spatials2], dim=0)
        image_mask = torch.stack([image_mask1, image_mask2], dim=0)
        caption = torch.stack([caption1, caption2], dim=0)
        input_mask = torch.stack([input_mask1, input_mask2], dim=0)
        segment_ids = torch.stack([segment_ids1, segment_ids2], dim=0)
        image_id = torch.from_numpy(np.array([image_id, image_id2]))
        target = 0

        return (
            features,
            spatials,
            image_mask,
            caption,
            target,
            input_mask,
            segment_ids,
            image_id,
        )

    def __len__(self):
        return len(self.entries)

def load_annotationsVal(annotations_jsonpath):
    with jsonlines.open(annotations_jsonpath) as reader:
        caption_entries = []
        for annotation in reader:
            image_id = int(annotation["img_path"].split(".")[0])

            sents = []
            for sentences in annotation["sentences"]:
                sents.append(sentences)
            caption_entries.append({"caption": sents, "image_id": image_id})
    return caption_entries

class RetrievalDatasetVal(Dataset):
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
        max_seq_length=20,
        max_region_num=51
    ):
        self.caption_entries = load_annotationsVal(anno_path)
        self.feat_folder = feat_folder
        self.tokenizer = tokenizer

        self.split = split
        self.padding_index = padding_index
        self.max_region_num = max_region_num
        self.max_seq_length = max_seq_length
        self.num_labels = 1

        cache_path = os.path.join(
            dataroot,
            "cache",
            task_name + "_" + split + "_" + str(max_seq_length) + ".pkl",
        )

        if not os.path.exists(cache_path):
            self.tokenize()
            cPickle.dump(self.caption_entries, open(cache_path, "wb"))
        else:
            self.caption_entries = cPickle.load(open(cache_path, "rb"))

    def tokenize(self):
        for entry in self.caption_entries:
            token_arr = []
            input_mask_arr = []
            segment_ids_arr = []
            for caption in entry["caption"]:
                tokens = self.tokenizer.encode(caption)
                tokens = tokens[: self.max_seq_length - 2]
                tokens = self.tokenizer.add_special_tokens_single_sentence(tokens)

                segment_ids = [0] * len(tokens)
                input_mask = [1] * len(tokens)

                if len(tokens) < self.max_seq_length:
                    padding = [self.padding_index] * (self.max_seq_length - len(tokens))
                    tokens = tokens + padding
                    input_mask += padding
                    segment_ids += padding

                assert_eq(len(tokens), self.max_seq_length)

                token_arr.append(tokens)
                input_mask_arr.append(input_mask)
                segment_ids_arr.append(segment_ids)

            entry["token"] = token_arr
            entry["input_mask"] = input_mask_arr
            entry["segment_ids"] = segment_ids_arr

    def load_feat_sent(self, entry):
        image_id = entry["image_id"]
        features, num_boxes, boxes = utils.image_features_reader(self.feat_folder, image_id)

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

        caption_arr = []
        input_mask_arr = []
        segment_ids_arr = []
        for i in range(len(entry["token"])):
            caption_arr.append(torch.from_numpy(np.array(entry["token"][i])))
            input_mask_arr.append(torch.from_numpy(np.array(entry["input_mask"][i])))
            segment_ids_arr.append(torch.from_numpy(np.array(entry["segment_ids"][i])))

        caption = torch.stack(caption_arr, dim=0)
        input_mask = torch.stack(input_mask_arr, dim=0)
        segment_ids = torch.stack(segment_ids_arr, dim=0)

        return features, image_mask, spatials, caption, input_mask, segment_ids

    def __getitem__(self, index):
        entry = self.caption_entries[index]
        image_id = entry["image_id"]

        features, image_mask, spatials, caption, input_mask, segment_ids = self.load_feat_sent(entry)
        cap_image_id = np.array(len(caption) * [image_id])

        return (
            features,
            spatials,
            image_mask,
            caption,
            input_mask,
            segment_ids,
            cap_image_id,
            image_id
        )

    def __len__(self):
        return len(self.caption_entries)
