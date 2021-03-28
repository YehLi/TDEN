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

def _converId(img_id):
    img_id = img_id.split("-")
    if "train" in img_id[0]:
        new_id = int(img_id[1])
    elif "val" in img_id[0]:
        new_id = int(img_id[1])
    elif "test" in img_id[0]:
        new_id = int(img_id[1])
    else:
        pdb.set_trace()
    return new_id


def load_annotationsQ_A(annotations_jsonpath, split):
    entries = []
    with open(annotations_jsonpath, "rb") as f:
        for annotation in json_lines.reader(f):
            question = annotation["question"]
            if split == "test":
                ans_label = 0
            else:
                ans_label = annotation["answer_label"]

            img_id = _converId(annotation["img_id"])
            img_fn = annotation["img_fn"]
            anno_id = int(annotation["annot_id"].split("-")[1])
            entries.append(
                {
                    "question": question,
                    "img_fn": img_fn,
                    "objects":  annotation["objects"],
                    "answers": annotation["answer_choices"],
                    "metadata_fn": annotation["metadata_fn"],
                    "target": ans_label,
                    "img_id": img_id,
                    "anno_id": anno_id,
                }
            )
    return entries

def load_annotationsQA_R(annotations_jsonpath, split):
    entries = []
    with open(annotations_jsonpath, "rb") as f:
        for annotation in json_lines.reader(f):
            if split == "test":
                for answer in annotation["answer_choices"]:
                    question = annotation["question"] + ["[SEP]"] + answer
                    img_id = _converId(annotation["img_id"])
                    ans_label = 0
                    img_fn = annotation["img_fn"]
                    anno_id = int(annotation["annot_id"].split("-")[1])
                    entries.append(
                        {
                            "question": question,
                            "img_fn": img_fn,
                            "objects":  annotation["objects"],
                            "answers": annotation["rationale_choices"],
                            "metadata_fn": annotation["metadata_fn"],
                            "target": ans_label,
                            "img_id": img_id,
                        }
                    )
            else:
                question = annotation["question"]
                ans_label = annotation["rationale_label"]
                img_id = _converId(annotation["img_id"])
                img_fn = annotation["img_fn"]

                anno_id = int(annotation["annot_id"].split("-")[1])
                entries.append(
                    {
                        "question": question,
                        "question_a": ["[SEP]"] + annotation["answer_choices"][annotation["answer_label"]],
                        "img_fn": img_fn,
                        "objects":  annotation["objects"],
                        "answers": annotation["rationale_choices"],
                        "metadata_fn": annotation["metadata_fn"],
                        "target": ans_label,
                        "img_id": img_id,
                        "anno_id": anno_id,
                    }
                )
    return entries

class VCRDataset(Dataset):
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
        if task_name == "VCR_Q-A":
            self.entries = load_annotationsQ_A(anno_path, split)
        elif task_name == "VCR_QA-R":
            self.entries = load_annotationsQA_R(anno_path, split)
        else:
            assert False
        self.split = split
        self.feat_folder = feat_folder
        self.gt_feat_folder = gt_feat_folder
        self.tokenizer = tokenizer
        self.task_name = task_name

        self.padding_index = padding_index
        self.max_caption_length = max_seq_length
        self.max_region_num = max_region_num
        self.num_labels = 1
        self.dataroot = dataroot

        self.names = ['Casey', 'Riley', 'Jessie', 'Jackie', 'Avery', 'Jaime', 'Peyton', 'Kerry', 'Jody', 'Kendall',
                        'Frankie', 'Pat', 'Quinn']

        cache_path = os.path.join(
            dataroot, "cache",
            task_name + "_" + split + "_"
            + str(max_seq_length) + "_" + str(max_region_num)
            + "_vcr_fn.pkl",
        )

        if not os.path.exists(cache_path):
            self.tokenize()
            cPickle.dump(self.entries, open(cache_path, "wb"))
        else:
            self.entries = cPickle.load(open(cache_path, "rb"))

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

            tokens_q, tags_q = self.retokenize_and_convert_to_ids_with_tag(entry["question"], objects_replace_name, self.non_obj_tag)
            if self.task_name == "VCR_QA-R":
                tokens_q2, tags_q2 = self.retokenize_and_convert_to_ids_with_tag(entry["question_a"], objects_replace_name, self.non_obj_tag)

            input_ids_all = []
            input_mask_all = []
            segment_ids_all = []
            tags_all = []

            for answer in entry["answers"]:
                tokens_r, tags_r = self.retokenize_and_convert_to_ids_with_tag(answer, objects_replace_name, self.non_obj_tag)

                if self.task_name == "VCR_Q-A":
                    tokens_q_copy = copy.copy(tokens_q)
                    tags_q_copy = copy.copy(tags_q)
                    self.truncate_seq_pair(tokens_q_copy, tags_q_copy, tokens_r, tags_r , self.max_caption_length - 3)
                else:
                    tokens_q_copy = copy.copy(tokens_q)
                    tags_q_copy = copy.copy(tags_q)
                    tokens_q2_copy = copy.copy(tokens_q2)
                    tags_q2_copy = copy.copy(tags_q2)
                    self.truncate_seq_tri(tokens_q_copy, tags_q_copy, tokens_q2_copy, tags_q2_copy, tokens_r, tags_r , self.max_caption_length - 3)
                    tokens_q_copy = tokens_q_copy + tokens_q2_copy
                    tags_q_copy = tags_q_copy + tags_q2_copy

                segment_ids = [0] * (len(tokens_q_copy) + 2) + [1] * (len(tokens_r) + 1)
                input_ids = self.tokenizer.add_special_tokens_sentences_pair(tokens_q_copy, tokens_r)
                tags_ids = [self.non_obj_tag] + tags_q_copy + [self.non_obj_tag] + tags_r + [self.non_obj_tag]
                input_mask = [1] * len(input_ids)

                while len(input_ids) < self.max_caption_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)
                    tags_ids.append(self.non_obj_tag)

                assert len(input_ids) == self.max_caption_length
                assert len(input_mask) == self.max_caption_length
                assert len(segment_ids) == self.max_caption_length
                assert len(tags_ids) == self.max_caption_length

                input_ids_all.append(input_ids)
                input_mask_all.append(input_mask)
                segment_ids_all.append(segment_ids)
                tags_all.append(tags_ids)

            entry["input_ids"] = input_ids_all
            entry["input_mask"] = input_mask_all
            entry["segment_ids"] = segment_ids_all
            entry["tag_ids"] = tags_all

    def retokenize_and_convert_to_ids_with_tag(self, _tokens, objects_replace_name, non_obj_tag=-1):
        tags = []
        parsed_tokens = []
        for mixed_token in _tokens:
            if isinstance(mixed_token, list):
                tokens = [objects_replace_name[o] for o in mixed_token]
                retokenized_tokens = self.tokenizer.tokenize(tokens[0])
                tags.extend([mixed_token[0] + non_obj_tag + 1 for _ in retokenized_tokens])
                for token, o in zip(tokens[1:], mixed_token[1:]):
                    retokenized_tokens.append('and')
                    tags.append(non_obj_tag)
                    
                    re_tokens = self.tokenizer.tokenize(token)
                    retokenized_tokens.extend(re_tokens)
                    tags.extend([o + non_obj_tag + 1 for _ in re_tokens])
                parsed_tokens.extend(retokenized_tokens)
            else:
                retokenized_tokens = self.tokenizer.tokenize(mixed_token)
                parsed_tokens.extend(retokenized_tokens)
                tags.extend([non_obj_tag for _ in retokenized_tokens])

        ids = self.tokenizer.convert_tokens_to_ids(parsed_tokens)
        return ids, tags
    
    def truncate_seq_pair(self, tokens_q, tags_q, tokens_a, tags_a, max_length):
        while len(tokens_a) + len(tokens_q) > max_length:
            if len(tokens_a) > len(tokens_q):
                tokens_a.pop()
                tags_a.pop()
            else:
                tokens_q.pop()
                tags_q.pop()

    def truncate_seq_tri(self, tokens_q, tags_q, tokens_a, tags_a, tokens_r, tags_r, max_length):
        while len(tokens_q) + len(tokens_a) + len(tokens_r) > max_length:
            if len(tokens_r) > (len(tokens_q) + len(tokens_a)):
                tokens_r.pop()
                tags_r.pop()
            elif len(tokens_q) > 1:
                tokens_q.pop()
                tags_q.pop()
            else:
                tokens_a.pop()
                tags_a.pop()

    def __getitem__(self, index):
        entry = self.entries[index]

        image_id = entry["img_id"]
        img_query = entry["metadata_fn"][:-5]

        prob = random.random()
        if prob > 0.5 and self.split == 'train':
            features, num_boxes, boxes = utils.image_features_reader(self.feat_folder + "_mirror", img_query)
        else:
            features, num_boxes, boxes = utils.image_features_reader(self.feat_folder, img_query)

        if prob > 0.5 and self.split == 'train':
            gt_features, gt_num_boxes, gt_boxes = utils.image_features_reader(self.gt_feat_folder + "_mirror", img_query)
        else:
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
        target = int(entry["target"])

        if self.split == "test":
            anno_id = entry["anno_id"]
        else:
            anno_id = entry["anno_id"]

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

    def __len__(self):
        return len(self.entries)
