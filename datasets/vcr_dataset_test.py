import json
from typing import Any, Dict, List
import random
import os

import torch
from torch.utils.data import Dataset
import numpy as np
import _pickle as cPickle
import json_lines
import copy
import pdb
import csv
import sys
import lib.utils as utils

def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)

def load_annotations(annotations_jsonpath):
    entries = []
    with open(annotations_jsonpath, "rb") as f:
         for annotation in json_lines.reader(f):
            anno_id = int(annotation["annot_id"].split("-")[1])
            entries.append(
                {
                    "anno_id": anno_id,
                    "metadata_fn": annotation["metadata_fn"],
                    "objects":  annotation["objects"],
                    "question": annotation["question"],
                    "answers": annotation["answer_choices"],
                    "rationale": annotation["rationale_choices"],
                }
            )
    return entries

class VCRTestDataset(Dataset):
    def __init__(
        self,
        task_name,
        dataroot,
        anno_path,
        split,
        feat_folder,
        gt_feat_folder,
        tokenizer,
        padding_index = 0,
        max_seq_length = 40,
        max_region_num = 80,
    ):
        self.entries = load_annotations(os.path.join(dataroot, 'test.jsonl'))
        self.feat_folder = feat_folder
        self.gt_feat_folder = gt_feat_folder
        self.tokenizer = tokenizer
        self.task_name = task_name

        self.padding_index = padding_index
        self.max_caption_length = 66
        self.max_region_num = max_region_num
        self.num_labels = 1
        self.dataroot = dataroot

        self.names = ['Casey', 'Riley', 'Jessie', 'Jackie', 'Avery', 'Jaime', 'Peyton', 'Kerry', 'Jody', 'Kendall',
            'Frankie', 'Pat', 'Quinn']

        cache_path = os.path.join(
            dataroot,
            "cache",
            task_name + "_" + split + "_"  + str(max_seq_length) + "_" + str(max_region_num) + "_vcr_test.pkl"
        )

        if not os.path.exists(cache_path):
            self.tokenize()
            cPickle.dump(self.entries, open(cache_path, "wb"))
        else:
            self.entries = cPickle.load(open(cache_path, "rb"))

    def __len__(self):
        return len(self.entries)

    def retokenize_and_convert_to_ids(self, _tokens, objects_replace_name):
        parsed_tokens = []
        for mixed_token in _tokens:
            if isinstance(mixed_token, list):
                tokens = [objects_replace_name[o] for o in mixed_token]
                retokenized_tokens = self.tokenizer.tokenize(tokens[0])
                for token in tokens[1:]:
                    retokenized_tokens.append('and')
                    re_tokens = self.tokenizer.tokenize(token)
                    retokenized_tokens.extend(re_tokens)

                parsed_tokens.extend(retokenized_tokens)
            else:
                retokenized_tokens = self.tokenizer.tokenize(mixed_token)
                parsed_tokens.extend(retokenized_tokens)

        ids = self.tokenizer.convert_tokens_to_ids(parsed_tokens)
        return ids

    def _truncate_seq_pair(self, tokens_q, tokens_a, max_length):
        while len(tokens_a) + len(tokens_q) > max_length:
            if len(tokens_a) > len(tokens_q):
                tokens_a.pop()
            else:
                tokens_q.pop()

    def _truncate_seq_tri(self, tokens_q, tokens_a, tokens_r, max_length):
        while len(tokens_q) + len(tokens_a) + len(tokens_r) > max_length:
            if len(tokens_r) > (len(tokens_q) + len(tokens_a)):
                tokens_r.pop()
            elif len(tokens_q) > 1:
                tokens_q.pop()
            else:
                tokens_a.pop()

    def tokenize(self):
        person_name_id = 0
        for entry in self.entries:
            objects_replace_name = []
            for o in entry['objects']:
                if o == 'person':
                    objects_replace_name.append(self.names[person_name_id])
                    person_name_id = (person_name_id + 1) % len(self.names)
                else:
                    objects_replace_name.append(o)

            tokens_q = self.retokenize_and_convert_to_ids(entry["question"], objects_replace_name)
            tokens_a_list = []
            tokens_r_list = []

            for answer in entry["answers"]: 
                tokens_a = self.retokenize_and_convert_to_ids(answer, objects_replace_name)
                tokens_a_list.append(tokens_a)

            for rationale in entry["rationale"]: 
                tokens_r = self.retokenize_and_convert_to_ids(rationale, objects_replace_name)
                tokens_r_list.append(tokens_r)

            input_ids_all = []
            input_mask_all = []
            segment_ids_all = []

            # VCR_Q-A
            for tokens_a in tokens_a_list:
                tokens_q_copy = copy.copy(tokens_q)
                tokens_a_copy = copy.copy(tokens_a)
                self._truncate_seq_pair(tokens_q_copy, tokens_a_copy, self.max_caption_length - 3)

                segment_ids = [0] * (len(tokens_q_copy) + 2) + [1] * (len(tokens_a_copy) + 1)
                input_ids = self.tokenizer.add_special_tokens_sentences_pair(tokens_q_copy, tokens_a_copy)
                input_mask = [1] * len(input_ids)

                while len(input_ids) < self.max_caption_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)

                assert len(input_ids) == self.max_caption_length
                assert len(input_mask) == self.max_caption_length
                assert len(segment_ids) == self.max_caption_length

                input_ids_all.append(input_ids)
                input_mask_all.append(input_mask)
                segment_ids_all.append(segment_ids)

            # VCR_QA-R
            for tokens_a in tokens_a_list:
                for tokens_r in tokens_r_list:
                    tokens_q_copy = copy.copy(tokens_q)
                    tokens_a_copy = copy.copy(tokens_a)
                    tokens_r_copy = copy.copy(tokens_r)

                    tokens_a_copy = [self.tokenizer.sep_token_id] + tokens_a_copy
                    self._truncate_seq_tri(tokens_q_copy, tokens_a_copy, tokens_r_copy, self.max_caption_length - 3)

                    tokens_qa_copy = tokens_q_copy + tokens_a_copy

                    segment_ids = [0] * (len(tokens_qa_copy) + 2) + [1] * (len(tokens_r_copy) + 1)
                    input_ids = self.tokenizer.add_special_tokens_sentences_pair(tokens_qa_copy, tokens_r_copy)
                    input_mask = [1] * len(input_ids)

                    while len(input_ids) < self.max_caption_length:
                        input_ids.append(0)
                        input_mask.append(0)
                        segment_ids.append(0)

                    assert len(input_ids) == self.max_caption_length
                    assert len(input_mask) == self.max_caption_length
                    assert len(segment_ids) == self.max_caption_length

                    input_ids_all.append(input_ids)
                    input_mask_all.append(input_mask)
                    segment_ids_all.append(segment_ids)

            assert len(input_ids_all) == 20
            assert len(input_mask_all) == 20
            assert len(segment_ids_all) == 20

            entry["input_ids"] = input_ids_all
            entry["input_mask"] = input_mask_all
            entry["segment_ids"] = segment_ids_all

    def __getitem__(self, index):
        entry = self.entries[index]
        anno_id = entry["anno_id"]
        img_query = entry["metadata_fn"][:-5]
        features, num_boxes, boxes = utils.image_features_reader(self.feat_folder, img_query)
        gt_features, gt_num_boxes, gt_boxes = utils.image_features_reader(self.gt_feat_folder, img_query)
        
        # merge two features.
        features[0] = (features[0] * num_boxes + gt_features[0] * gt_num_boxes) / (
            num_boxes + gt_num_boxes
        )

        # merge two boxes, and assign the labels.
        gt_boxes = gt_boxes[1:gt_num_boxes]
        gt_features = gt_features[1:gt_num_boxes]
        gt_num_boxes = gt_num_boxes - 1

        gt_box_preserve = min(self.max_region_num - 1, gt_num_boxes)
        gt_boxes = gt_boxes[:gt_box_preserve]
        gt_features = gt_features[:gt_box_preserve]
        gt_num_boxes = gt_box_preserve

        num_box_preserve = min(self.max_region_num - int(gt_num_boxes), int(num_boxes))
        boxes = boxes[:num_box_preserve]
        features = features[:num_box_preserve]

        # concatenate the boxes
        mix_boxes = np.concatenate((boxes, gt_boxes), axis=0)
        mix_features = np.concatenate((features, gt_features), axis=0)
        mix_num_boxes = num_box_preserve + int(gt_num_boxes)

        image_mask = [1] * (mix_num_boxes)
        while len(image_mask) < self.max_region_num:
            image_mask.append(0)

        mix_boxes_pad = np.zeros((self.max_region_num, 5))
        mix_features_pad = np.zeros((self.max_region_num, 2048))

        mix_boxes_pad[:mix_num_boxes] = mix_boxes[:mix_num_boxes]
        mix_features_pad[:mix_num_boxes] = mix_features[:mix_num_boxes]

        # appending the target feature.
        features = torch.tensor(mix_features_pad).float()
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(mix_boxes_pad).float()

        input_ids = torch.from_numpy(np.array(entry["input_ids"]))
        input_mask = torch.from_numpy(np.array(entry["input_mask"]))
        segment_ids = torch.from_numpy(np.array(entry["segment_ids"]))
        target = 0

        return (
            features,
            spatials,
            image_mask,
            input_ids,
            target,
            input_mask,
            segment_ids,
            anno_id,
        )


