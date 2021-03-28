from .vcr_dataset import VCRDataset
from .vcr_dataset_test import VCRTestDataset
from .vqa_dataset import VQAClassificationDataset
from .retrieval_dataset import RetrievalDataset, RetrievalDatasetVal
from .coco_cap_dataset import COCOCapDataset, COCOTestCapDataset

DatasetMapTrain = {
    "VQA": VQAClassificationDataset,
    "VCR_Q-A": VCRDataset,
    "VCR_QA-R": VCRDataset,
    "RetrievalFlickr30k": RetrievalDataset,
    'Caption': COCOCapDataset,
    'VCR_TEST': VCRTestDataset
}

DatasetMapEval = {
    "VQA": VQAClassificationDataset,
    "VCR_Q-A": VCRDataset,
    "VCR_QA-R": VCRDataset,
    "RetrievalFlickr30k": RetrievalDatasetVal,
    'Caption': COCOCapDataset,
    'VCR_TEST': VCRTestDataset
}