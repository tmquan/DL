import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscriminativeLoss(nn.Module):
    def __init__(self, delta_v, delta_d, alpha=1.0, beta=1.0, gamma=0.001):
        super(DiscriminativeLoss, self).__init__()
        self.delta_d = delta_d
        self.delta_v = delta_v
        self.gamma = gamma
        self.beta  = beta
        self.alpha = alpha

    def _var_loss(self, embed, label, delta_v):
        pass

    def _dis_loss(self, embed, label, delta_d):
        pass

    def _reg_loss(self):
        pass

    def forward(self, embeds, labels):
        var_loss = self._var_loss(embeds, labels, self.delta_v)
        dis_loss = self._dis_loss(embeds, labels, self.delta_d)
        reg_loss = self._reg_loss(embeds) 

        loss = self.alpha*var_loss + self.beta*dis_loss + self.gamma*reg_loss
        return loss