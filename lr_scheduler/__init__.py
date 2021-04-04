from lr_scheduler.multi_step_lr import MultiStepLR

__factory = {
    'MultiStep': MultiStepLR,
}

def names():
    return sorted(__factory.keys())

def create(name, optimizer,  *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown lr_scheduler:", name)
    return __factory[name](optimizer, *args, **kwargs)
