import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class SemanticLoss(nn.Module):
    def __init__(self, target_size = (640, 640)):
        super().__init__()
    
        self.target_size = target_size

    def forward(self, logits, target):
        # """
        # returns weighted sum: focal_w * Focal + dice_w * Dice
        # """
        target_resized = target.unsqueeze(1).float()
        target_resized = F.interpolate(target_resized, size=self.target_size, mode='nearest').squeeze(1).long()  # B x h_out x w_out

        loss_sem = F.cross_entropy(logits, target_resized)
        return loss_sem, torch.tensor(0.0), torch.tensor(0.0)