import os
import sys
import csv
import copy
import json
import pickle
import random
import numpy as np
import json_lines
import pdb
import _pickle as cPickle
import lib.utils as utils
from lib.config import cfg
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

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
    """Build an index out of FOIL annotations, mapping each image ID with its corresponding captions."""
    entries = []
    with open(annotations_jsonpath, "rb") as f:  # opening file in binary(rb) mode
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
    """Build an index out of FOIL annotations, mapping each image ID with its corresponding captions."""
    entries = []
    with open(annotations_jsonpath, "rb") as f:  # opening file in binary(rb) mode
        for annotation in json_lines.reader(f):
            if split == "test":
                # for each answer
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

def iou(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.shape[0]
    K = gt_boxes.shape[0]

    gt_boxes_area = (
        (gt_boxes[:, 2] - gt_boxes[:, 0] + 1) * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)
    ).reshape(1, K)

    anchors_area = (
        (anchors[:, 2] - anchors[:, 0] + 1) * (anchors[:, 3] - anchors[:, 1] + 1)
    ).reshape(N, 1)

    boxes = np.repeat(anchors.reshape(N, 1, 4), K, axis=1)
    query_boxes = np.repeat(gt_boxes.reshape(1, K, 4), N, axis=0)

    iw = (
        np.minimum(boxes[:, :, 2], query_boxes[:, :, 2])
        - np.maximum(boxes[:, :, 0], query_boxes[:, :, 0])
        + 1
    )
    iw[iw < 0] = 0

    ih = (
        np.minimum(boxes[:, :, 3], query_boxes[:, :, 3])
        - np.maximum(boxes[:, :, 1], query_boxes[:, :, 1])
        + 1
    )
    ih[ih < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    return overlaps

class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(
        self, 
        image_feat=None,
        image_target=None,
        caption=None,
        lm_labels=None,
        image_loc=None,
        num_boxes=None,
        overlaps=None,
        is_mask=None,
        segment_ids=None
    ):
        self.image_feat = image_feat
        self.caption = caption
        self.lm_labels = lm_labels  # masked words for language model
        self.image_loc = image_loc
        self.image_target = image_target
        self.num_boxes = num_boxes
        self.overlaps = overlaps
        self.is_mask = is_mask
        self.segment_ids = segment_ids

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self,
        input_ids=None,
        input_mask=None,
        segment_ids=None,
        is_match=None,
        lm_label_ids=None,
        vfeat=None,
        vtarget=None,
        vlabel=None,
        vmask=None
    ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.is_match = is_match
        self.lm_label_ids = lm_label_ids
        self.vfeat = vfeat
        self.vtarget = vtarget
        self.vlabel = vlabel
        self.vmask = vmask

class VCRPretrainDataset(Dataset):
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
        max_region_num = 80
    ):
        self.entries_q2a = load_annotationsQ_A(anno_path, split)
        self.entries_qa2r = load_annotationsQA_R(anno_path, split)

        self.split = split
        self.feat_folder = feat_folder
        self.gt_feat_folder = gt_feat_folder
        self.tokenizer = tokenizer

        self.padding_index = padding_index
        self.max_caption_length = max_seq_length
        self.max_region_num = max_region_num
        self.num_labels = 1
        self.dataroot = dataroot
        self.person_name_id = 0

        self._names = ['Casey', 'Riley', 'Jessie', 'Jackie', 'Avery', 'Jaime', 'Peyton', 'Kerry', 'Jody', 'Kendall',
            'Frankie', 'Pat', 'Quinn']

    def __len__(self):
        return len(self.entries_q2a)

    def image_features_reader(self, feat_folder, feat_folder_gt, image_id):
        #image_id_ = 565546
        content_gt = np.load(os.path.join(feat_folder_gt, str(image_id) + '.npz'))
        features_gt = content_gt['features']
        cls_prob_gt = content_gt['cls_prob']
        num_boxes_gt = content_gt['num_boxes'][0]
        boxes_gt = content_gt['boxes']
        image_h = content_gt['image_h'][0]
        image_w = content_gt['image_w'][0]

        content_p = np.load(os.path.join(feat_folder, str(image_id) + '.npz'))
        features_p = content_p['features']
        cls_prob_p = content_p['cls_prob']
        num_boxes_p = content_p['num_boxes'][0]
        boxes_p = content_p['boxes']

        num_boxes = num_boxes_gt + num_boxes_p
        features = np.concatenate([features_gt, features_p], axis=0)
        cls_prob = np.concatenate([cls_prob_gt, cls_prob_p], axis=0)
        boxes = np.concatenate([boxes_gt, boxes_p], axis=0)

        # calculate the IOU here.
        overlaps = iou(boxes, boxes)

        image_location = np.zeros((boxes.shape[0], 5), dtype=np.float32)
        image_location[:, :4] = boxes
        image_location[:, 4] = (
            (image_location[:, 3] - image_location[:, 1])
            * (image_location[:, 2] - image_location[:, 0])
            / (float(image_w) * float(image_h))
        )

        image_location[:, 0] = image_location[:, 0] / float(image_w)
        image_location[:, 1] = image_location[:, 1] / float(image_h)
        image_location[:, 2] = image_location[:, 2] / float(image_w)
        image_location[:, 3] = image_location[:, 3] / float(image_h)

        return features, cls_prob, num_boxes, image_location, overlaps

    def random_word(self, tokens, is_mask, tokenizer):
        output_label = []

        for i, token in enumerate(tokens):
            if is_mask[i] == 0:
                output_label.append(-1)
                continue

            prob = random.random()
            # mask token with 15% probability

            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = np.random.randint(len(tokenizer))
                    # torch.randint(len(tokenizer), labels.shape, dtype=torch.long)

                # -> rest 10% randomly keep current token
                # append current token to output (we will predict these later)
                output_label.append(token)
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)

        return tokens, output_label

    def random_region(self, image_feat, image_loc, num_boxes, overlaps):
        output_label = []
        masked_label = np.zeros((image_feat.shape[0]))

        for i in range(num_boxes):
            prob = random.random()
            # mask token with 15% probability

            # if is_next == 1 and self.objective != 0:
            #     prob = 1 # if the target is inaligned mask, then not sample mask
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.9:
                    image_feat[i] = 0
                # mask the overlap regions into zeros
                masked_label = np.logical_or(masked_label, overlaps[i] >= 1.0)

                # 10% randomly change token to random token
                # elif prob < 0.9:
                # tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

                # -> rest 10% randomly keep current token
                # append current token to output (we will predict these later)
                output_label.append(1)
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)

        masked_label = [idx for idx, item in enumerate(masked_label) if item]
        if masked_label:
            image_feat[masked_label, :] = 0

        return image_feat, image_loc, output_label, masked_label

    def retokenize_and_convert_to_ids_with_tag(self, _tokens, objects_replace_name):
        is_mask = []
        parsed_tokens = []
        for mixed_token in _tokens:
            if isinstance(mixed_token, list):
                tokens = [objects_replace_name[o] for o in mixed_token]
                retokenized_tokens = self.tokenizer.tokenize(tokens[0])
                is_mask.extend([0 for _ in retokenized_tokens])
                for token, o in zip(tokens[1:], mixed_token[1:]):
                    retokenized_tokens.append('and')
                    is_mask.append(0)
                    
                    re_tokens = self.tokenizer.tokenize(token)
                    retokenized_tokens.extend(re_tokens)
                    is_mask.extend([0 for _ in re_tokens])
                parsed_tokens.extend(retokenized_tokens)
            else:
                retokenized_tokens = self.tokenizer.tokenize(mixed_token)
                parsed_tokens.extend(retokenized_tokens)
                is_mask.extend([1 for _ in retokenized_tokens])

        ids = self.tokenizer.convert_tokens_to_ids(parsed_tokens)
        assert len(ids) == len(is_mask)
        return ids, is_mask

    def _truncate_seq_pair(self, tokens_q, tags_q, tokens_a, tags_a, max_length):
        while len(tokens_a) + len(tokens_q) > max_length:
            if len(tokens_a) > len(tokens_q):
                tokens_a.pop()
                tags_a.pop()
            else:
                tokens_q.pop()
                tags_q.pop()

    def _truncate_seq_tri(self, tokens_q, tags_q, tokens_a, tags_a, tokens_r, tags_r, max_length):
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

    def get_caption(self, entry, task_name):
        objects_replace_name = []
        for o in entry['objects']:
            if o == 'person':
                objects_replace_name.append(self._names[self.person_name_id])
                self.person_name_id = (self.person_name_id + 1) % len(self._names)
            else:
                objects_replace_name.append(o)

        tokens_q, is_mask = self.retokenize_and_convert_to_ids_with_tag(entry["question"], objects_replace_name)
        if task_name == "VCR_QA-R":
            tokens_q2, is_mask2 = self.retokenize_and_convert_to_ids_with_tag(entry["question_a"], objects_replace_name)
            is_mask2[0] = 0

        target = int(entry["target"])
        ans_id = random.sample(range(len(entry["answers"])), 1)[0]
        is_match = [1.0] if target == ans_id else [0.0]
        answer = entry["answers"][ans_id]

        tokens_r, is_mask_r = self.retokenize_and_convert_to_ids_with_tag(answer, objects_replace_name)

        if task_name == "VCR_Q-A":
            tokens_q_copy = copy.copy(tokens_q)
            is_mask_copy = copy.copy(is_mask)
            self._truncate_seq_pair(tokens_q_copy, is_mask_copy, tokens_r, is_mask_r , self.max_caption_length - 3)
        else:
            tokens_q_copy = copy.copy(tokens_q)
            is_mask_copy = copy.copy(is_mask)
            tokens_q2_copy = copy.copy(tokens_q2)
            is_mask2_copy = copy.copy(is_mask2)
            self._truncate_seq_tri(tokens_q_copy, is_mask_copy, tokens_q2_copy, is_mask2_copy, tokens_r, is_mask_r , self.max_caption_length - 3)
            tokens_q_copy = tokens_q_copy + tokens_q2_copy
            is_mask_copy = is_mask_copy + is_mask2_copy

        segment_ids = [0] * (len(tokens_q_copy) + 2) + [1] * (len(tokens_r) + 1)
        tokens = self.tokenizer.add_special_tokens_sentences_pair(tokens_q_copy, tokens_r)
        is_mask = [0] + is_mask_copy + [0] + is_mask_r + [0]
        assert len(is_mask) == len(tokens)
        return tokens, is_mask, segment_ids, is_match

    def convert_example_to_features(
        self, example, max_seq_length, tokenizer, max_region_num
    ):
        image_feat = example.image_feat
        tokens = example.caption
        image_loc = example.image_loc
        image_target = example.image_target
        num_boxes = int(example.num_boxes)
        overlaps = example.overlaps
        is_mask = example.is_mask
        segment_ids = example.segment_ids

        tokens, tokens_label = self.random_word(tokens, is_mask, tokenizer)
        lm_label_ids = tokens_label
        input_mask = [1] * len(tokens)

        input_ids = tokens
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            lm_label_ids.append(-1)
            input_mask.append(0)
            segment_ids.append(0)

        mix_num_boxes = min(int(num_boxes), max_region_num)
        mix_boxes_pad = np.zeros((max_region_num, 5))
        mix_features_pad = np.zeros((max_region_num, 2048))
        mix_target_pad = np.zeros((max_region_num, image_target.shape[1]))

        image_feat, image_loc, image_label, masked_label = self.random_region(
            image_feat, image_loc, mix_num_boxes, overlaps
        )
        mix_boxes_pad[:mix_num_boxes] = image_loc[:mix_num_boxes]
        mix_features_pad[:mix_num_boxes] = image_feat[:mix_num_boxes]
        mix_target_pad[:mix_num_boxes] = image_target[:mix_num_boxes]
        image_label = image_label[:mix_num_boxes]

        image_mask = [1] * (int(mix_num_boxes))
        while len(image_mask) < max_region_num:
            image_mask.append(0)
            image_label.append(-1)

        input_ids = np.array(input_ids)
        input_mask = np.array(input_mask)
        segment_ids = np.array(segment_ids)
        lm_label_ids = np.array(lm_label_ids)
        image_label = np.array(image_label)

        return input_ids, input_mask, segment_ids, lm_label_ids, \
            mix_features_pad, mix_boxes_pad, mix_target_pad, \
            image_label, image_mask, masked_label


    def __getitem__(self, index):
        entry_q2a = self.entries_q2a[index]
        entry_qa2r = self.entries_qa2r[index]

        prob = random.random()
        if prob > 0.5:
            task_name = "VCR_Q-A"
            entry = entry_q2a
        else:
            task_name = "VCR_QA-R"
            entry = entry_qa2r

        anno_id = entry_q2a["anno_id"]
        img_query = entry_q2a["metadata_fn"][:-5]

        prob = random.random()
        if prob > 0.5:
            feat_folder = self.feat_folder + "_mirror"
            gt_feat_folder = self.gt_feat_folder + "_mirror"
        else:
            feat_folder = self.feat_folder
            gt_feat_folder = self.gt_feat_folder

        image_feature, image_target, num_boxes, image_location, overlaps = self.image_features_reader(feat_folder, gt_feat_folder, img_query)

        tokens_caption, is_mask, segment_ids, is_match = self.get_caption(entry, task_name)

        cur_example = InputExample(
            image_feat=image_feature,
            image_target=image_target,
            caption=tokens_caption,
            image_loc=image_location,
            num_boxes=num_boxes,
            overlaps=overlaps,
            is_mask=is_mask,
            segment_ids=segment_ids
        )

        input_ids, input_mask, segment_ids, lm_label_ids, \
        image_feat, image_loc, mix_target_pad,  \
        image_label, image_mask, masked_label = \
            self.convert_example_to_features(cur_example, self.max_caption_length, self.tokenizer, self.max_region_num - 1)

        mix_num_boxes = min(int(num_boxes), self.max_region_num - 1)
        sum_count = max(1, mix_num_boxes - len(masked_label))  
        g_image_feat = np.sum(image_feat, axis=0) / sum_count
        image_feat = np.concatenate([np.expand_dims(g_image_feat, axis=0), image_feat], axis=0)
        image_feat = np.array(image_feat, dtype=np.float32)

        g_image_loc = np.array([0, 0, 1, 1, 1])
        image_loc = np.concatenate([np.expand_dims(g_image_loc, axis=0), image_loc], axis=0)
        image_loc = np.array(image_loc, dtype=np.float32)

        mix_target_pad = np.array(mix_target_pad, dtype=np.float32)

        image_mask = [1] + image_mask
        image_mask = np.array(image_mask)
        is_match = np.array(is_match, dtype=np.float32)

        batch = (
            input_ids,
            input_mask,
            segment_ids,
            lm_label_ids,
            image_feat,
            image_loc,
            mix_target_pad,
            image_label,
            image_mask,
            is_match
        )

        return batch