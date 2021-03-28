import os
import json
import _pickle as cPickle
import logging
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import lib.utils as utils

def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)

def create_entry(question, answer):
    answer.pop("image_id")
    answer.pop("question_id")
    entry = {
        "question_id": question["question_id"],
        "image_id": question["image_id"],
        "question": question["question"],
        "answer": answer,
    }
    return entry


def load_dataset(dataroot, name):
    if name == "trainval":
        question_path_train = os.path.join(dataroot, "v2_OpenEnded_mscoco_%s2014_questions.json" % "train")
        questions_train = sorted(
            json.load(open(question_path_train))["questions"],
            key=lambda x: x["question_id"],
        )
        answer_path_train = os.path.join(dataroot, "cache", "%s_target.pkl" % "train")
        answers_train = cPickle.load(open(answer_path_train, "rb"))
        answers_train = sorted(answers_train, key=lambda x: x["question_id"])

        question_path_val = os.path.join(dataroot, "v2_OpenEnded_mscoco_%s2014_questions.json" % "val")
        questions_val = sorted(
            json.load(open(question_path_val))["questions"],
            key=lambda x: x["question_id"],
        )   
        answer_path_val = os.path.join(dataroot, "cache", "%s_target.pkl" % "val")
        answers_val = cPickle.load(open(answer_path_val, "rb"))
        answers_val = sorted(answers_val, key=lambda x: x["question_id"])

        # VG
        vg_question_path_train = os.path.join(dataroot, "VG_questions2.json")
        vg_questions_train = sorted(
            json.load(open(vg_question_path_train))["questions"],
            key=lambda x: x["question_id"],
        )
        vg_answer_path_train = os.path.join(dataroot, "cache", "%s_target.pkl" % "vg")
        vg_answers_train = cPickle.load(open(vg_answer_path_train, "rb"))
        vg_answers_train = sorted(vg_answers_train, key=lambda x: x["question_id"])

        questions = questions_train + questions_val[:-3000] + vg_questions_train
        answers = answers_train + answers_val[:-3000] + vg_answers_train

    elif name == "minval":
        question_path_val = os.path.join(dataroot, "v2_OpenEnded_mscoco_%s2014_questions.json" % "val")
        questions_val = sorted(
            json.load(open(question_path_val))["questions"],
            key=lambda x: x["question_id"],
        )
        answer_path_val = os.path.join(dataroot, "cache", "%s_target.pkl" % "val")
        answers_val = cPickle.load(open(answer_path_val, "rb"))
        answers_val = sorted(answers_val, key=lambda x: x["question_id"])
        questions = questions_val[-3000:]
        answers = answers_val[-3000:]

    elif name == "test":
        question_path_test = os.path.join(dataroot, "v2_OpenEnded_mscoco_%s2015_questions.json" % name)
        questions_test = sorted(
            json.load(open(question_path_test))["questions"],
            key=lambda x: x["question_id"],
        )
        questions = questions_test
    else:
        assert False, "data split is not recognized."

    if "test" in name:
        entries = []
        for question in questions:
            entries.append(question)
    else:
        assert_eq(len(questions), len(answers))
        entries = []
        for question, answer in zip(questions, answers):
            assert_eq(question["question_id"], answer["question_id"])
            assert_eq(question["image_id"], answer["image_id"])
            entries.append(create_entry(question, answer))

    return entries

class VQAClassificationDataset(Dataset):
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
        max_region_num=51,
    ):
        super().__init__()
        self.split = split
        ans2label_path = os.path.join(dataroot, "cache", "trainval_ans2label.pkl")
        label2ans_path = os.path.join(dataroot, "cache", "trainval_label2ans.pkl")
        self.ans2label = cPickle.load(open(ans2label_path, "rb"))
        self.label2ans = cPickle.load(open(label2ans_path, "rb"))
        self.num_labels = len(self.ans2label)
        self.max_region_num = max_region_num
        self.max_seq_length = max_seq_length
        self.feat_folder = feat_folder
        self.tokenizer = tokenizer
        self.padding_index = padding_index

        if 'test' in self.split:
            pos = self.feat_folder.rfind('/')
            self.feat_folder = os.path.join(self.feat_folder[:pos], 'test2015_' + self.feat_folder[pos+1:])

        if self.split == 'minval':
            answers_val = cPickle.load(open(os.path.join(dataroot, "cache", "%s_target.pkl" % "val"), "rb"))
            self.id2datum = {}
            for datum in answers_val:
                quesid = datum['question_id']
                self.id2datum[quesid] = {}
                for i, label in enumerate(datum['labels']):
                    label_str = self.label2ans[label]
                    self.id2datum[quesid][label_str] = datum['scores'][i]

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
            tokens = self.tokenizer.encode(entry["question"])
            tokens = tokens[: max_length - 2]
            tokens = self.tokenizer.add_special_tokens_single_sentence(tokens)

            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)

            if len(tokens) < max_length:
                padding = [self.padding_index] * (max_length - len(tokens))
                tokens = tokens + padding
                input_mask += padding
                segment_ids += padding

            assert_eq(len(tokens), max_length)
            entry["q_token"] = tokens
            entry["q_input_mask"] = input_mask
            entry["q_segment_ids"] = segment_ids
            entry.pop("question")

    def __getitem__(self, index):
        entry = self.entries[index]
        image_id = entry["image_id"]
        question_id = entry["question_id"]

        prob = random.random()
        if prob > 0.5 and self.split == 'trainval':
            features, num_boxes, boxes = utils.image_features_reader(self.feat_folder + "_mirror", image_id)
        else:
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

        question = np.array(entry["q_token"])
        input_mask = np.array(entry["q_input_mask"])
        segment_ids = np.array(entry["q_segment_ids"])
        target = torch.zeros(self.num_labels)

        ############### exchange right & left if mirror ###################
        if prob > 0.5 and self.split == 'trainval':
            for i in range(1, len(question)):
                if question[i] == 2187:
                    question[i] = 2157
                elif question[i] == 2157:
                    question[i] = 2187

        if "test" not in self.split:
            answer = entry["answer"]
            labels = np.array(answer["labels"])
            scores = np.array(answer["scores"], dtype=np.float32)
            if prob > 0.5 and self.split == 'trainval':
                for i in range(len(labels)):
                    if labels[i] == self.ans2label['left']:
                        labels[i] = self.ans2label['right']
                    elif labels[i] == self.ans2label['right']:
                        labels[i] = self.ans2label['left']
            
            if len(labels) == 0:
                labels = None
                scores = None
            else:
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)

            if labels is not None:
                target.scatter_(0, labels, scores)

        return (
            features,
            spatials,
            image_mask,
            question,
            target,
            input_mask,
            segment_ids,
            question_id,
        )

    def __len__(self):
        return len(self.entries)
