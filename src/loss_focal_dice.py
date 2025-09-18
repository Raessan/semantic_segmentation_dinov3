import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def softmax_focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction='mean'):
    """
    inputs: (N, C, H, W) raw logits
    targets: (N, H, W) with class indices (0..C-1)
    """
    log_probs = F.log_softmax(inputs, dim=1)  # (N, C, H, W)
    probs = torch.exp(log_probs)
    
    targets = targets.long()
    ce_loss = F.nll_loss(log_probs, targets, reduction='none')  # (N, H, W)

    # Gather probabilities of the true class
    pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (N, H, W)

    focal_term = (1 - pt) ** gamma
    loss = focal_term * ce_loss

    if alpha is not None:
        alpha_factor = torch.full_like(inputs, alpha)
        alpha_factor = alpha_factor.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss *= alpha_factor

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss

def multiclass_dice_loss(inputs, targets, reduction='mean'):
    """
    inputs: (N, C, H, W) raw logits
    targets: (N, H, W) with class indices (0..C-1)
    """
    num_classes = inputs.shape[1]
    inputs = F.softmax(inputs, dim=1)  # (N, C, H, W)
    targets_onehot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

    inputs = inputs.flatten(2)  # (N, C, H*W)
    targets_onehot = targets_onehot.flatten(2)

    numerator = 2 * (inputs * targets_onehot).sum(-1)
    denominator = inputs.sum(-1) + targets_onehot.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss

class SemanticLoss(nn.Module):
    def __init__(self, dice_weight=1.0, focal_weight=1.0,
                 focal_gamma=2.0, focal_alpha=None, target_size = (640, 640), ignore_index=None):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.target_size = target_size

    def forward(self, logits, targets):
        # """
        # returns weighted sum: focal_w * Focal + dice_w * Dice
        # """
        targets_resized = targets.unsqueeze(1).float()
        targets_resized = F.interpolate(targets_resized, size=self.target_size, mode='nearest').squeeze(1).long()  # B x h_out x w_out

        f = softmax_focal_loss(logits, targets_resized, 
                               alpha=self.focal_alpha, 
                               gamma=self.focal_gamma, 
                               reduction='mean')
        d = multiclass_dice_loss(logits, targets_resized, reduction='mean')
        return self.focal_weight * f + self.dice_weight * d, self.focal_weight*f, self.dice_weight*d
    
if __name__ == "__main__":
    torch.manual_seed(0)

    # --- toy shapes and params ---
    B, C = 2, 134                      # batch size and num classes
    H_lab, W_lab = 640, 640        
    target_hw = (320, 320)           

    # Fake model logits at 320x320 (match target_size expected by the loss)
    logits = torch.randn(B, C, target_hw[0], target_hw[1], requires_grad=True)

    # Fake integer targets (class ids in [0, C-1]) at a different size
    targets = torch.randint(0, C, (B, H_lab, W_lab), dtype=torch.long)

    # Instantiate the criterion
    criterion = SemanticLoss(
        dice_weight=1.0,
        focal_weight=1.0,
        focal_gamma=2.0,
        focal_alpha=0.25,
        target_size=target_hw      
    )

    # Compute losses
    total_loss, focal_loss_val, dice_loss_val = criterion(logits, targets)

    print(
        f"total: {total_loss.item():.4f} | "
        f"focal: {focal_loss_val.item():.4f} | "
        f"dice: {dice_loss_val.item():.4f}"
    )