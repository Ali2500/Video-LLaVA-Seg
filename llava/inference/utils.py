from typing import List, Dict, Any, Optional
from llava.vision_utils import get_resize_padding_params

import torch
import torch.nn.functional as F
import numpy as np
import einops
import pycocotools.mask as mt


def preprocess_seg_inputs(
        seg_frames: torch.Tensor, 
        seg_meta: Dict[str, Any], 
        video_processor, 
        tgt_size: int, 
        normalize: bool,
        seg_masks: Optional[torch.Tensor] = None,
        pad_mode: Optional[str] = "topleft"
    ):
    # seg_frames: [T, C, H, W] (uint8), 0-255
    # seg_masks: [T, N, H, W] or None
    seg_frames = seg_frames.to(torch.float32)

    # if video_processor.do_rescale:
    seg_frames = seg_frames / 255.

    if video_processor.do_normalize and normalize:
        img_mean = torch.tensor(video_processor.image_mean, dtype=torch.float32)[None, :, None, None]
        img_std = torch.tensor(video_processor.image_std, dtype=torch.float32)[None, :, None, None]
        seg_frames = (seg_frames - img_mean) / img_std

    if seg_masks is not None:
        assert seg_frames.shape[-2:] == seg_masks.shape[-2:]

    # resize to larger dim == target size, and then center pad the shorter dim
    h, w = seg_frames.shape[-2:]
    (h, w), (pad_left, pad_right, pad_top, pad_bottom) = get_resize_padding_params(h, w, tgt_size, pad_mode=pad_mode)
    # pad_left = pad_right = pad_top = pad_bottom = 0

    # if h > w:
    #     h = tgt_size
    #     w = int(round((w / h) * tgt_size))
    #     pad_left = (h - w) // 2
    #     pad_right = h - w - pad_left
    # else:
    #     w = tgt_size
    #     h = int(round((h / w) * tgt_size))
    #     pad_top = (w - h) // 2
    #     pad_bottom = w - h - pad_top

    seg_frames = F.interpolate(seg_frames, (h, w), mode='bilinear', align_corners=False)
    seg_frames = F.pad(seg_frames, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

    if seg_masks is not None:
        dtype = seg_masks.dtype  # F.interpolate does not work with bool dtype
        assert dtype in (torch.uint8, torch.bool)
        seg_masks = seg_masks.byte()
        seg_masks = F.interpolate(seg_masks, (h, w), mode='nearest-exact')
        seg_masks = F.pad(seg_masks, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        seg_masks = seg_masks.to(dtype)

    seg_meta["resized_image_size"] = (h, w)
    seg_meta["padding"] = (pad_left, pad_right, pad_top, pad_bottom)

    if seg_masks is None:
        return seg_frames, seg_meta
    else:
        return seg_frames, seg_masks, seg_meta
    

def mask_tensor_to_rle(mask_tensor: torch.Tensor, decode_utf8: bool = True) -> List[List[Dict[str, Any]]]:
    # mask_tensor: [N, T, H, W]
    assert mask_tensor.dtype in (torch.bool, torch.uint8)
    mask_tensor = mask_tensor.numpy().astype(np.uint8)

    rle_list = []
    for masks_per_track in mask_tensor:
        rle_list.append([])
        for mask_per_frame in masks_per_track:
            rle = mt.encode(np.asfortranarray(mask_per_frame))
            if decode_utf8:
                rle["counts"] = rle["counts"].decode("utf-8")
            rle_list[-1].append(rle)

    return rle_list


def get_null_mask_rle(height: int, width: int, decode_utf8: bool = True):
    mask = np.zeros(height, width, np.uint8)
    rle = mt.encode(np.asfortranarray(mask))
    if decode_utf8:
        rle["counts"] = rle["counts"].decode("utf-8")
    return rle
