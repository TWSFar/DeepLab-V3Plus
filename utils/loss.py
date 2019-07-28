import torch
import torch.nn as nn


class SegmentationLoss(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda
    
    def build_loss(self, mode='ce'):
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        else:
            raise NotImplementedError
        
    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        crti

