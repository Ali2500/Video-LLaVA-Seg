# The functions in this module have been copied verbatim from:
# https://github.com/Ali2500/TarViS/blob/main/tarvis/data/utils/image_cropping.py
# and 
# https://github.com/Ali2500/TarViS/blob/main/tarvis/data/utils/video_cropping.py

from typing import List, Tuple, Optional, Union
from torch import Tensor

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import random


def mask_to_bbox(masks: Tensor, raise_error_if_null_mask: Optional[bool] = True) -> torch.Tensor:
    """
    Extracts bounding boxes from masks
    :param masks: tensor of shape [N_1, ..., N_d, H, W]
    :param raise_error_if_null_mask: Flag for whether or not to raise an error if a mask is all-zeros.
    :return: tensor of shape [N, 4] containing bounding boxes coordinates in [x, y, w, h] format.
             If `raise_error_if_null_mask` is False, coordinates [-1, -1, -1, -1] will be returned for all-zeros masks.
    """
    assert masks.ndim > 2

    # flatten additional leading dims
    leading_dim_sizes = masks.shape[:-2]
    masks = masks.reshape(-1, *masks.shape[-2:])  # [N, H, W]
    assert masks.ndim == 3  # sanity check

    null_masks = torch.logical_not(torch.any(masks.flatten(1), 1))[:, None]  # [N, 1]
    if torch.any(null_masks) and raise_error_if_null_mask:
        raise ValueError("One or more all-zero masks found")

    h, w = masks.shape[-2:]

    reduced_rows = torch.any(masks, 2).long()  # [N, H]
    reduced_cols = torch.any(masks, 1).long()  # [N, W]

    x_min = (reduced_cols * torch.arange(-w-1, -1, dtype=torch.long, device=masks.device)[None]).argmin(1)  # [N]
    y_min = (reduced_rows * torch.arange(-h-1, -1, dtype=torch.long, device=masks.device)[None]).argmin(1)  # [N]

    x_max = (reduced_cols * torch.arange(w, dtype=torch.long, device=masks.device)[None]).argmax(1)  # [N]
    y_max = (reduced_rows * torch.arange(h, dtype=torch.long, device=masks.device)[None]).argmax(1)  # [N]

    width = x_max - x_min + 1
    height = y_max - y_min + 1

    bbox_coords = torch.stack((x_min, y_min, width, height), 1)
    invalid_box = torch.full_like(bbox_coords, -1)

    bbox_coords = torch.where(null_masks, invalid_box, bbox_coords)  # [N, 4]
    return bbox_coords.reshape(*leading_dim_sizes, 4)  # [..., 4]


def compute_mask_containing_image_crop(
        mask: Union[np.ndarray, Tensor],
        target_dims: Tuple[int, int],
        min_crop_mask_coverage: float = 1e-8,
        num_tries: int = 10,
        containment_criterion: Optional[str] = "nonzero"
    ) -> Union[Tuple[int, int], None]:
    """
    Computes crop parameters for resizing the given `mask` to `target_dims` such that AT LEAST SOME nonzero points in
    `mask` are contained in the crop.
    :param mask: tensor of numpy array of shape [H, W] with non-zero values for active pixels
    :param target_dims: Desired size of crop as (height, width) tuple
    :param min_crop_mask_coverage: fraction of active pixels which must be part of the final crop. Valid range: [0, 1).
                                E.g. if 0.5 is given, then the returned crop will contain at least 50% of the active
                                pixels contained in the input mask
    :param num_tries: number of attempts made before returning None
    :param containment_criterion:
    :return:
    """
    assert mask.ndim == 2
    assert 0 < min_crop_mask_coverage <= 1.0
    assert num_tries > 0
    assert containment_criterion in ("nonzero", "unique_values")

    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(np.ascontiguousarray(mask))

    x1, y1, box_w, box_h = mask_to_bbox(mask.unsqueeze(0), raise_error_if_null_mask=True)[0].tolist()
    x2 = x1 + box_w
    y2 = y1 + box_h

    im_height, im_width = mask.shape[-2:]
    crop_height, crop_width = target_dims
    num_required_crop_mask_points = ((mask > 0).sum(dtype=torch.float32) * min_crop_mask_coverage).clamp(min=1.0).int().item()

    x1_min = max(0, x1 - crop_width + 1)
    x1_max = min(im_width - crop_width, x2 - 1)
    # print("x range: ", x1_min, x1_max, x1, x2, y1, y2)
    assert x1_max >= x1_min, f"Invalid range for values of x1 for crop: [{x1_min}, {x1_max}]. Box dims: [{x1}, {y1}, {x2}, {y2}]. " \
        f"Image dims: [{im_height}, {im_width}]. Crop dims: [{crop_height}, {crop_width}]"

    y1_min = max(0, y1 - crop_height + 1)
    y1_max = min(im_height - crop_height, y2 - 1)
    assert y1_max >= y1_min, f"Invalid range for values of y1 for crop: [{y1_min}, {y1_max}]. Box dims: [{x1}, {y1}, {x2}, {y2}]. " \
        f"Image dims: [{im_height}, {im_width}]. Crop dims: [{crop_height}, {crop_width}]"

    valid_crop_x1, valid_crop_y1 = -1, -1

    # This is a bit harder than computing a mask preserving crop. What we can do is ensure that at least part of the
    # bounding box enclosed by the active mask region is included in the crop, but this actually does not guarantee
    # that an active pixel will be part of the crop. The easy solution is to just try this multiple (num_tries) times
    # and hope that we get a valid crop.
    for _ in range(num_tries):
        # sample x1 and y1 for crop
        crop_x1 = random.randint(x1_min, x1_max)
        crop_y1 = random.randint(y1_min, y1_max)

        mask_crop = mask[crop_y1:crop_y1+crop_height, crop_x1:crop_x1+crop_width]

        if containment_criterion == "nonzero":
            if (mask_crop > 0).sum(dtype=torch.long).item() >= num_required_crop_mask_points:
                valid_crop_x1, valid_crop_y1 = crop_x1, crop_y1
                break

        if containment_criterion == "unique_values":
            input_mask_values, input_mask_counts = mask.unique(sorted=True, return_counts=True)
            crop_mask_values, crop_mask_counts = mask_crop.unique(sorted=True, return_counts=True)

            if input_mask_values.numel() != crop_mask_values.numel():
                continue

            coverage_ratios = crop_mask_counts.float() / input_mask_counts.float()
            if torch.all(coverage_ratios >= min_crop_mask_coverage):
                valid_crop_x1, valid_crop_y1 = crop_x1, crop_y1
                break

    if valid_crop_x1 >= 0:
        return valid_crop_x1, valid_crop_y1
    else:
        return None


def compute_mask_containing_video_crop(
        mask: Union[np.ndarray, Tensor],
        target_dims: Tuple[int, int],
        min_crop_mask_coverage: float = 0.25,
        num_tries: int = 10,
        containment_criterion: Optional[str] = "nonzero"
    ) -> Union[Tuple[int, int], None]:
    """
    Computes crop parameters for resizing the given video `mask` to `target_dims` such that AT LEAST SOME nonzero
    points in `mask` are contained in the crop. Note that the crop is intended to be the same across the temporal
    dimension
    :param mask: tensor of numpy array of shape [T, H, W] with non-zero values for active pixels
    :param target_dims: Desired size of crop as (height, width) tuple
    :param num_tries: number of attempts made before returning None
    :param min_crop_mask_coverage
    :param containment_criterion
    :return:
    """
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(np.ascontiguousarray(mask))

    # collapse the mask along the time dimension and then just compute a mask containing image crop
    mask = torch.any(mask, 0)  # [H, W]
    return compute_mask_containing_image_crop(
        mask, target_dims, min_crop_mask_coverage=min_crop_mask_coverage, num_tries=num_tries,
        containment_criterion=containment_criterion
    )
