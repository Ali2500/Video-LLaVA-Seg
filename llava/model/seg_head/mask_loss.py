from einops import rearrange
from typing import List
from torch import Tensor

from llava.model.seg_head.hungarian_matcher import HungarianMatcher

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.matcher = HungarianMatcher(cost_mask=1, cost_dice=1, num_points=12544)
        self.weight_dice = 1.0 # 0.5
        self.weight_ce = 2.0

    def forward(self, pred_masks: List[Tensor], gt_masks: List[Tensor], dummy_forward_pass: bool):
        """
        pred_masks: List of length batch size, each element being a tensor of shape [N, T, H, W]. Unscaled logits.
        gt_masks: List of length batch size, each element being a tensor of shape [N, T, H, W]. Dtype bool.
        """
        pred_masks_all, gt_masks_all = [], []
        pred_ious = []

        for pred_masks_b, gt_masks_b in zip(pred_masks, gt_masks):
            assert pred_masks_b.shape == gt_masks_b.shape, f"Shape mismatch: {pred_masks_b.shape} =/= {gt_masks_b.shape}"
            num_objs = pred_masks_b.shape[0]
            if num_objs > 1:
                # need to perform hungarian matching between gt and pred
                pred_masks_b, gt_masks_b = self.align_pred_and_gt(pred_masks_b, gt_masks_b)

            pred_ious.append(self.compute_iou(pred_masks_b, gt_masks_b))

            pred_masks_b = rearrange(pred_masks_b, "n t h w -> (n t) h w")
            gt_masks_b = rearrange(gt_masks_b, "n t h w -> (n t) h w")

            pred_masks_all.append(pred_masks_b)  # combine object and time dims
            gt_masks_all.append(gt_masks_b)

        pred_masks_all = torch.cat(pred_masks_all)
        gt_masks_all = torch.cat(gt_masks_all)
        pred_ious = torch.cat(pred_ious)

        # print(pred_masks_all.min().item(), pred_masks_all.max().item())
        num_masks = pred_masks_all.shape[0]
        pred_masks_all = pred_masks_all.float()
        gt_masks_all = gt_masks_all.to(pred_masks_all.dtype)

        if dummy_forward_pass:
            loss_dice = loss_ce = pred_masks_all.sum() * 0.0
            pred_ious = torch.tensor([]).to(pred_ious)
        else:
            loss_dice = self.dice_loss(inputs=pred_masks_all, targets=gt_masks_all, num_masks=num_masks)
            loss_ce = self.sigmoid_ce_loss(inputs=pred_masks_all, targets=gt_masks_all, num_masks=num_masks)

        # print(f"Internal: {loss_dice.item()}, {loss_ce.item()}")
        return {
            "loss_mask_dice": loss_dice,
            "loss_mask_ce": loss_ce,
            "loss_mask": (loss_dice * self.weight_dice) + (loss_ce * self.weight_ce),
            "mask_ious": pred_ious
        }        

    def align_pred_and_gt(self, pred_masks: Tensor, gt_masks: Tensor):
        # pred_masks: [N, T, H, W]
        # gt_masks: [N, T, H, W]

        gt_targets = [{
            "masks": gt_masks
        }]
        pred_outputs = {
            "pred_masks": pred_masks.unsqueeze(0)  # [1, N, T, H, W]
        }

        pred_idx, gt_idx = self.matcher(outputs=pred_outputs, targets=gt_targets)[0]
        pred_masks = pred_masks[pred_idx]
        gt_masks = gt_masks[gt_idx]

        return pred_masks, gt_masks

    def dice_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
        scale=1000,  # 100000.0,
        eps=1e-6,
    ):
        """
        Compute the DICE loss, similar to generalized IOU for masks
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        """
        assert inputs.ndim == targets.ndim == 3
        assert inputs.shape == targets.shape
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1, 2)  # [M, H, W] -> [M, H*W]
        targets = targets.flatten(1, 2)  # [M, H, W] -> [M, H*W]
        numerator = 2 * (inputs / scale * targets).sum(-1)
        denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
        loss = 1 - (numerator + eps) / (denominator + eps)
        loss = loss.sum() / (num_masks + 1e-8)
        return loss

    def sigmoid_ce_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
        """
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        Returns:
            Loss tensor
        """
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
        return loss

    @torch.no_grad()
    def compute_iou(self, pred_masks: torch.Tensor, gt_masks: torch.Tensor):
        # pred_masks: [B, T, H, W] (unscaled logits)
        # gt_masks: [B, T, H, W] (bool)
        assert pred_masks.shape == gt_masks.shape, f"Shape mismatch: {pred_masks.shape}, {gt_masks.shape}"
        scale = 1.0
        pred_masks = pred_masks > 0.0
        pred_masks = pred_masks.flatten(1)
        gt_masks = gt_masks.flatten(1)
        intersection = (torch.logical_and(pred_masks, gt_masks).to(torch.float32) / scale).sum(1)  # [B]
        union = (torch.logical_or(pred_masks, gt_masks).to(torch.float32) / scale).sum(1)  # [B]
        return intersection / union.clamp(min=1e-4)
