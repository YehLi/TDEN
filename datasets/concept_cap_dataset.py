import os
import sys
import csv
import copy
import pickle
import random
import numpy as np
import torch
import torch.utils.data as data
from lib.config import cfg

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
        image_loc=None,
        num_boxes=None,
        overlaps=None,
    ):
        self.image_feat = image_feat
        self.caption = caption
        self.image_loc = image_loc
        self.image_target = image_target
        self.num_boxes = num_boxes
        self.overlaps = overlaps

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

class ConceptCap(data.Dataset):
    def __init__(
        self,
        task_name,
        dataroot,
        anno_file,
        phrase_file,
        feat_folder,
        tokenizer,
        bert_model,
        padding_index=0,
        max_seq_length=16,
        max_region_num=51
    ):
        self.feat_folder = feat_folder
        self.tokenizer = tokenizer
        self.padding_index = padding_index
        self.max_seq_length = max_seq_length
        self.max_region_num = max_region_num
        self.mask_token_id = cfg.MODEL.MASK_ID
        self.sep_token_id = cfg.MODEL.SEP_ID

        cap_path = os.path.join(dataroot, anno_file)           # Train_GCC-training.tsv
        feat_path = os.path.join(dataroot, 'image_path.txt')

        self.captions = {}
        with open(cap_path) as f:
            rd = csv.reader(f, delimiter='\t', quotechar='"')
            image_id = 1
            for row in rd:
                self.captions[image_id] = row[0]
                image_id += 1
            
        with open(feat_path, 'r') as fid:
            self.feat_paths = [line.strip() for line in fid]
        self.image_ids = [int(item.split('/')[-1].split('.')[0]) for item in self.feat_paths]

        print('Find %d features' % int(len(self.feat_paths)))

    def __len__(self):
        return len(self.feat_paths)

    def image_features_reader(self, feat_folder, image_id):
        content = np.load(os.path.join(feat_folder, str(image_id)))
        features = content['features']
        cls_prob = content['cls_prob']
        num_boxes = content['num_boxes'][0]
        boxes = content['boxes']
        image_h = content['image_h'][0]
        image_w = content['image_w'][0]

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

    def random_word(self, tokens, tokenizer):
        output_label = []

        for i, token in enumerate(tokens):
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
                masked_label = np.logical_or(masked_label, overlaps[i] > 0.4)

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


    def convert_example_to_features(
        self, example, max_seq_length, tokenizer, max_region_num
    ):
        image_feat = example.image_feat
        tokens = example.caption
        image_loc = example.image_loc
        image_target = example.image_target
        num_boxes = int(example.num_boxes)
        overlaps = example.overlaps

        tokens = tokens[: max_seq_length - 2]
        tokens, tokens_label = self.random_word(tokens, tokenizer)
        lm_label_ids = [-1] + tokens_label + [-1]
        tokens = tokenizer.add_special_tokens_single_sentence(tokens)
        segment_ids = [0] * max_seq_length + [1] * max_seq_length
        input_mask = torch.tril(torch.ones(
            (max_seq_length, max_seq_length), dtype=torch.long))
        mask_len = len(tokens)
        input_mask[:, mask_len:] = 0

        input_ids = tokens
        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            lm_label_ids.append(-1)

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
        image_id = self.image_ids[index]
        caption = self.captions[image_id]
        image_path = self.feat_paths[index]

        image_feature, image_target, num_boxes, image_location, overlaps = self.image_features_reader(self.feat_folder, image_path)  
        tokens_caption = self.tokenizer.encode(caption)

        cur_example = InputExample(
            image_feat=image_feature,
            image_target=image_target,
            caption=tokens_caption,
            image_loc=image_location,
            num_boxes=num_boxes,
            overlaps=overlaps
        )

        input_ids, input_mask, segment_ids, lm_label_ids, \
        image_feat, image_loc, mix_target_pad,  \
        image_label, image_mask, masked_label = \
            self.convert_example_to_features(cur_example, self.max_seq_length, self.tokenizer, self.max_region_num - 1)

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

        batch = (
            input_ids,
            input_mask,
            segment_ids,
            lm_label_ids,
            image_feat,
            image_loc,
            mix_target_pad,
            image_label,
            image_mask
        )
        
        return batch

    

    

    

    
