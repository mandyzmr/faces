from torch import optim
from config import *

def get_optimizer(model, name='sgd'):
    if name=='sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    elif name=='adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    elif name=='adagrad':
        optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()),
                      lr =LEARNING_RATE, lr_decay=1e-4, weight_decay=WEIGHT_DECAY)
    return optimizer


def adjust_learning_rate(optimizer, epoch):
    if epoch<2:
        lr = 1e-4
    elif 2<=epoch<4:
        lr = 1e-5
    elif 4<=epoch<6:
        lr = 1e-6
    elif 6<=epoch<10:
        lr = 1e-7
    elif epoch>=10:
        lr = 1e-8
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

