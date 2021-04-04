import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lib.config import cfg
import lr_scheduler
import lr_scheduler.warmup_lr as warmup_lr
from optimizer.bertadam import BertAdam

class Optimizer(nn.Module):
    def __init__(self, model, epoch_steps=1, n_steps=0):
        super(Optimizer, self).__init__()

        params = []
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue
            lr = cfg.SOLVER.LEARNING_RATE
            weight_decay = cfg.SOLVER.WEIGHT_DECAY

            if any(nd in key for nd in no_decay):
                weight_decay = 0

            if 'bert.encoder.layer' in key:
                lr = lr * cfg.SOLVER.BERT_LR_FACTOR

            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        if cfg.SOLVER.TYPE == 'SGD':
            self.optimizer = torch.optim.SGD(params, 
                lr=cfg.SOLVER.LEARNING_RATE, 
                momentum=cfg.SOLVER.SGD.MOMENTUM)
        elif cfg.SOLVER.TYPE == 'ADAM':
            self.optimizer = torch.optim.Adam(params,
                lr=cfg.SOLVER.LEARNING_RATE, betas=cfg.SOLVER.ADAM.BETAS, eps=cfg.SOLVER.ADAM.EPS)
        elif cfg.SOLVER.TYPE == 'ADAMAX':
            self.optimizer = torch.optim.Adamax(params,
                lr=cfg.SOLVER.LEARNING_RATE, betas=cfg.SOLVER.ADAM.BETAS, eps=cfg.SOLVER.ADAM.EPS)
        elif cfg.SOLVER.TYPE == 'ADAGRAD':
            self.optimizer = torch.optim.Adagrad(params,
                lr=cfg.SOLVER.LEARNING_RATE)
        elif cfg.SOLVER.TYPE == 'RMSPROP':
            self.optimizer = torch.optim.RMSprop(params, 
                lr=cfg.SOLVER.LEARNING_RATE)
        elif cfg.SOLVER.TYPE == 'BERTADAM':
            self.optimizer =  BertAdam(
                params,
                warmup=cfg.SOLVER.WARMUP_PROPORTION,
                t_total=n_steps,
                schedule=cfg.SOLVER.LR_POLICY.BERTADAM_SCHEDULE)
        else:
            raise NotImplementedError

        warmup_steps = cfg.SOLVER.WARMUP_PROPORTION * n_steps
        if cfg.SOLVER.LR_POLICY.TYPE == 'Fix':
            self.scheduler = None
        elif cfg.SOLVER.LR_POLICY.TYPE == 'Step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=cfg.SOLVER.LR_POLICY.STEP_SIZE * epoch_steps, 
                gamma=cfg.SOLVER.LR_POLICY.GAMMA)
        elif cfg.SOLVER.LR_POLICY.TYPE == 'MultiStep':
            steps = [step * epoch_steps for step in cfg.SOLVER.LR_POLICY.STEPS]
            self.scheduler = lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=steps, 
                gamma=cfg.SOLVER.LR_POLICY.GAMMA)
        elif cfg.SOLVER.LR_POLICY.TYPE == 'warmup_constant':
            self.scheduler = warmup_lr.WarmupConstantSchedule(
                self.optimizer, 
                warmup_steps=warmup_steps)
        elif cfg.SOLVER.LR_POLICY.TYPE == 'warmup_linear':
            self.scheduler = warmup_lr.WarmupLinearSchedule(
                self.optimizer, 
                warmup_steps=warmup_steps,
                t_total=n_steps)
        elif cfg.SOLVER.LR_POLICY.TYPE == 'warmup_cosine':
            self.scheduler = warmup_lr.WarmupCosineSchedule(
                self.optimizer, 
                warmup_steps=warmup_steps,
                t_total=n_steps)
        elif cfg.SOLVER.LR_POLICY.TYPE == 'warmup_multistep':
            steps = [step * epoch_steps for step in cfg.SOLVER.LR_POLICY.STEPS]
            self.scheduler = warmup_lr.WarmupMultiStepLR(
                self.optimizer,
                milestones=steps,
                gamma=cfg.SOLVER.LR_POLICY.GAMMA,
                warmup_factor=0,
                warmup_iters=warmup_steps,
                warmup_method="linear",
            )
        
        if cfg.SOLVER.TYPE == 'BERTADAM' and cfg.SOLVER.LR_POLICY.BERTADAM_SCHEDULE != 'warmup_constant':
            self.scheduler = None

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self, inc_steps=None):
        if inc_steps is None:
            self.optimizer.step()
        else:
            self.optimizer.step(inc_steps=inc_steps)

    def scheduler_step(self, iters=None):
        if self.scheduler is not None:
            self.scheduler.step(iters)

    def get_lr(self):
        if cfg.SOLVER.TYPE == 'BERTADAM':
            return self.optimizer.show_lr()
        else:
            lr = []
            for param_group in self.optimizer.param_groups:
                lr.append(param_group['lr'])
            lr = sorted(list(set(lr)))
            return lr
