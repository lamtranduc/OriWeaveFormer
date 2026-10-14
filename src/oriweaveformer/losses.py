import torch
import torch.nn.functional as F


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor,
                  epsilon: float = 1e-7) -> torch.Tensor:
    """Numerically stable BCE plus batch-global soft Dice."""
    # Keep the unchanged BCE + Dice formula; large FP16 reductions can overflow.
    logits, target = logits.float(), target.float()
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum()
    dice = 1.0 - (2.0 * intersection + epsilon) / (
        probability.sum() + target.sum() + epsilon)
    return F.binary_cross_entropy_with_logits(logits, target) + dice
