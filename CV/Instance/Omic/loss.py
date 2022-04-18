import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscriminativeLoss(nn.Module):
    def __init__(self, delta_v=0.5, delta_d=1.5, alpha=1.0, beta=1.0, gamma=0.001):
        super(DiscriminativeLoss, self).__init__()
        self.delta_d = delta_d
        self.delta_v = delta_v
        self.gamma = gamma
        self.beta  = beta
        self.alpha = alpha

    def _cal_mean(self, embed, label, include_background=False):
        # This function will iterate over the batch
        # Find the unique value of each cluster in the label
        # Calculate the mean of each cluster in the embedding inclusively
        # Broadcast the mean of each cluster to the masked embedding 
        B, _, H, W = label.shape
        B, F, H, W = embed.shape
        means = embed.clone()
        for b in range(B):
            b_label = label[[b], :1, :, :]
            b_embed = embed[[b], :F, :, :]
            b_means = means[[b], :F, :, :]
            # Find the unique value of each cluster in the label, including background
            # b_value, b_count = torch.unique(b_label.view(1, -1), return_counts=True, dim=1)
            b_value = torch.unique(b_label.view(1, -1), dim=1)

            for value in b_value:
                b_mask_ = (value == b_label).expand(1, F, -1)
                # b_means = b_embed[b_mask_].means(axis=(-1, -2), keepdim=True)
                m_means = torch.masked_select(b_embed, b_mask_).view(1, F, -1).mean(dim=-1, keepdim=True) # 1 x F x S < HW
                b_means[b_mask_] = m_means

            # Update the cluster means
            means[[b], :F, :, :] = b_means
        return means 

    def _var_loss(self, embed, label, means, delta_v=0.5, include_background=False):
        var_loss = 0
        B, F, H, W = means.shape

        for b in range(B):
            b_label = label[[b], :1, :, :]
            b_embed = embed[[b], :F, :, :]
            b_means = means[[b], :F, :, :]
            b_value = torch.unique(b_label.view(1, -1), dim=1)

            C = len(b_value)

            if C>0:
                b_var_loss = 0
                # for c in range(C-1):
                    # b_var_loss += (torch.clamp(torch.norm(mu_c - x_c, dim=1) - 0.5, min=0) ** 2).mean()
                # b_var_loss += (torch.clamp(torch.norm(means - embed, dim=1) - delta_v, min=0) ** 2).mean()
                for value in b_value:
                    b_mask_ = (value == b_label).expand(1, F, -1)
                    # b_means = b_embed[b_mask_].means(axis=(-1, -2), keepdim=True)
                    m_means = torch.masked_select(b_embed, b_mask_).view(1, F, -1).mean(dim=-1, keepdim=True) # 1 x F x S < HW
                    m_embed = torch.masked_select(b_embed, b_mask_).view(1, F, -1) # 1 x F x S < HW
                    b_var_loss += (torch.clamp(torch.norm(m_means - m_embed, dim=1) - delta_v, min=0) ** 2).mean()
                b_var_loss /= C
            else:
                b_var_loss = 0

            var_loss += b_var_loss
        return var_loss

    def _dis_loss(self, embed, label, means, delta_d=1.5, include_background=False):
        dis_loss = 0
        B, F, H, W = means.shape

        for b in range(B):
            b_label = label[[b], :1, :, :]
            b_embed = embed[[b], :F, :, :]
            b_means = means[[b], :F, :, :]
            b_value = torch.unique(b_label.view(1, -1), dim=1)
            C = len(b_value)
            
            if C>2:
                b_dis_loss = 0
                for ca in range(C-1):
                    for cb in range(ca+1, C):
                        ma = b_value[ca]
                        mb = b_value[cb]
                        b_dis_loss += torch.clamp(delta_d - torch.norm(ma - mb), min=0)**2
                b_dis_loss /= (C*(C-1))
            else:
                b_dis_loss = 0

            dis_loss += b_dis_loss
        return dis_loss

    def _reg_loss(self, embed, label, means, include_background=False):
        reg_loss = 0
        B, _, H, W = label.shape
        B, F, H, W = embed.shape
        means = embed.clone()
        for b in range(B):
            b_label = label[[b], :1, :, :]
            b_embed = embed[[b], :F, :, :]
            b_means = means[[b], :F, :, :]
            # Find the unique value of each cluster in the label, including background
            # b_value, b_count = torch.unique(b_label.view(1, -1), return_counts=True, dim=1)
            b_value = torch.unique(b_label.view(1, -1), dim=1)

            C = len(b_value)

            if C>1:
                b_reg_loss = 0
                for value in b_value:
                    b_mask_ = (value == b_label).expand(1, F, -1)
                    # b_means = b_embed[b_mask_].means(axis=(-1, -2), keepdim=True)
                    m_means = torch.masked_select(b_embed, b_mask_).view(1, F, -1).mean(dim=-1, keepdim=False) # 1 x F x S < HW
                    b_reg_loss += torch.norm(m_means, dim=1).mean()
            else:
                b_reg_loss = 0
            
            reg_loss += b_reg_loss
        return reg_loss

    def forward(self, embeds, labels):
        means = self._cal_mean(embeds, labels, including_background=False)
        var_loss = self._var_loss(embeds, labels, means, delta_v=self.delta_v, include_background=False)
        dis_loss = self._dis_loss(embeds, labels, means, delta_d=self.delta_D, include_background=False)
        reg_loss = self._reg_loss(embeds, labels, means, include_background=False)

        loss = self.alpha*var_loss + self.beta*dis_loss + self.gamma*reg_loss
        return loss