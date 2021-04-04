import os
import os.path as osp
import numpy as np

from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from lib.config import cfg
cfg = __C

# ---------------------------------------------------------------------------- #
# Training options
# ---------------------------------------------------------------------------- #
__C.TRAIN = edict()

__C.TRAIN.BERT_MODEL = 'bert-base-uncased'

__C.TRAIN.FROM_PRETRAINED = 'bert-base-uncased'

__C.TRAIN.DO_LOWER_CASE = True

# ---------------------------------------------------------------------------- #
# PRETRAIN options
# ---------------------------------------------------------------------------- #
__C.PRETRAIN = edit()

__C.PRETRAIN.BATCH_SIZE = 512

__C.PRETRAIN.DATAROOT = ''

__C.PRETRAIN.ANNO = ''

__C.PRETRAIN.FEAT_FOLDER = ''

__C.PRETRAIN.MAX_SEQ_LEN = 36

__C.PRETRAIN.MAX_REGION_NUM = 51

# ---------------------------------------------------------------------------- #
# Data loader options
# ---------------------------------------------------------------------------- #
__C.DATA_LOADER = edict()

__C.DATA_LOADER.NUM_WORKERS = 4

__C.DATA_LOADER.USE_CHUNK = 0

__C.DATA_LOADER.PIN_MEMORY = True

__C.DATA_LOADER.DROP_LAST = True

# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #
__C.MODEL = edict()

__C.MODEL.BERT_ENCODE = '' 

__C.MODEL.CLS_ID = ''

__C.MODEL.SEP_ID = ''

__C.MODEL.MASK_ID = ''

# ---------------------------------------------------------------------------- #
# Solver options
# ---------------------------------------------------------------------------- #
__C.SOLVER = edict()

# Solver type
__C.SOLVER.TYPE = 'BertAdam'                 # 'ADAM', 'ADAMAX', 'SGD', 'ADAGRAD', 'RMSPROP'

__C.SOLVER.LEARNING_RATE = 0.0001

__C.SOLVER.WEIGHT_DECAY = 0.01

__C.SOLVER.BERT_LR_FACTOR = 1.0

__C.SOLVER.START_EPOCH = 0

__C.SOLVER.NUM_TRAIN_EPOCHS = 10

__C.SOLVER.WARMUP_PROPORTION = 0.1

__C.SOLVER.DISPLAY = 100

__C.SOLVER.TEST_INTERVAL = 1

__C.SOLVER.SNAPSHOT_ITERS = 1

# SGD
__C.SOLVER.SGD = edict()
__C.SOLVER.SGD.MOMENTUM = 0.9

# ADAM
__C.SOLVER.ADAM = edict()
__C.SOLVER.ADAM.BETAS = [0.9, 0.999]
__C.SOLVER.ADAM.EPS = 1e-8

# LR_POLICY
# Schedule type (see functions in utils.lr_policy for options)
# E.g., 'step', 'steps_with_decay', ...
__C.SOLVER.LR_POLICY = edict()
__C.SOLVER.LR_POLICY.TYPE = 'Step'       # 'Fix', 'Step', 'MultiStep', 'Poly', Noam'
__C.SOLVER.LR_POLICY.GAMMA = 0.8         # For 'step', the current LR is multiplied by SOLVER.GAMMA at each step
__C.SOLVER.LR_POLICY.STEP_SIZE = 3       # Uniform step size for 'steps' policy
__C.SOLVER.LR_POLICY.STEPS = (3,)        # Non-uniform step iterations for 'steps_with_decay' or 'steps_with_lrs' policies
__C.SOLVER.LR_POLICY.BERTADAM_SCHEDULE = 'warmup_linear' # warmup_constant  warmup_linear
__C.SOLVER.CLIP_GRAD = -1.0
__C.SOLVER.GRAD_CLIP_TYPE = 'Norm'      # 'Clamp', 'Norm'

# ---------------------------------------------------------------------------- #
# TASK
# ---------------------------------------------------------------------------- #
__C.TASK = edict()
__C.TASK.SEL = [0, 1, 2, 3]
__C.TASK.NAME = ['VQA']
__C.TASK.TYPE = ['VL-classifier']
__C.TASK.LOSS = ['BCEWithLogitLoss']
__C.TASK.PROCESS = ['normal']
__C.TASK.DATAROOT = [''] 
__C.TASK.FEAT_FOLDER = ['']
__C.TASK.GT_FEAT_FOLDER = ['']
__C.TASK.TRAIN_ANNO = ['']
__C.TASK.VAL_ANNO = ['']
__C.TASK.MAX_SEQ_LEN = [23]
__C.TASK.MAX_REGION_NUM = [101]
__C.TASK.BATCH_SIZE = [128]
__C.TASK.EVAL_BATCH_SIZE = [1024] 
__C.TASK.TRAIN_SPLIT = ['trainval']
__C.TASK.VAL_SPLIT = ['minval']


# ---------------------------------------------------------------------------- #
# Losses options
# ---------------------------------------------------------------------------- #
__C.LOSSES = edict()

__C.LOSSES.MARGIN = 0.5

__C.LOSSES.LABELSMOOTHING = 0.1

__C.LOSSES.MAX_VIOLATION = False

# ---------------------------------------------------------------------------- #
# SCORER options
# ---------------------------------------------------------------------------- #
__C.SCORER = edict()

__C.SCORER.TYPES = ['Cider']

__C.SCORER.WEIGHTS = [1.0]

__C.SCORER.GT_PATH = 'coco_train_gts.pkl'

__C.SCORER.CIDER_CACHED = 'coco_train_cider.pkl'

# ---------------------------------------------------------------------------- #
# Inference options
# ---------------------------------------------------------------------------- #
__C.INFERENCE = edict()

__C.INFERENCE.ANNFILE = 'captions_test5k.json'

__C.INFERENCE.BEAM_SIZE = 1

__C.INFERENCE.COCO_PATH = '../coco_caption'


__C.SEED = -1.0  # random seed for initialization

__C.LOGGER_NAME = 'log'  # Logger name

# Root directory of project
__C.ROOT_DIR = os.getcwd()

__C.CONFIG_FILE = 'config/bert_config.json'

__C.TEMP_DIR = './data/temp'



def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    #for k, v in a.iteritems(): python2
    for k, v in a.items(): # python3
        # a must specify keys that are in b
        #if not b.has_key(k):
        if not k in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                'for config key: {}').format(type(b[k]),
                                                            type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)
