import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from typing import List, Dict, Any
from torch import Tensor
from sam2.sam2_image_predictor import SAM2ImagePredictor


class SegmentationHeadSAM2(nn.Module):
    def __init__(self, n_token_dims: int, n_vision_dims: int, n_seg_queries: int, variant: str):
        super().__init__()
        # variant = "facebook/sam2.1-hiera-tiny" OR "facebook/sam2.1-hiera-small"  OR "facebook/sam2-hiera-base-plus"
        sam2 = SAM2ImagePredictor.from_pretrained(variant).model.to(torch.bfloat16)
 
        hidden_dims = 256
        self.n_seg_queries = n_seg_queries
        self.proj_token = nn.Linear(n_token_dims, hidden_dims * n_seg_queries)
    
        self.image_encoder = sam2.image_encoder
        self.prompt_encoder = sam2.sam_prompt_encoder
        self.mask_decoder = sam2.sam_mask_decoder
        self.no_mem_embed = sam2.no_mem_embed.permute(0, 2, 1)[:, :, :, None]  # [1, 1, 256] -> [1, 256, 1, 1]

        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    @property
    def has_image_encoder(self):
        return True

    def encode_video_frames(self, video_frames: Tensor):
        # video_frames: [B, 3, H, W]
        video_frames = (video_frames - self.image_mean) / self.image_std

        output = self.image_encoder(video_frames)
        high_res_features = [
            self.mask_decoder.conv_s0(output['backbone_fpn'][0]),  # [B, 256, H/4, W/4]
            self.mask_decoder.conv_s1(output['backbone_fpn'][1])  # [B, 256, H/8, W/8]
        ]
        backbone_feats = output["backbone_fpn"][2]  # [B, 256, H/16, W/16]

        # whether or not to do this depends on 'directly_add_no_mem_embed' in predictor.model
        backbone_feats = backbone_feats + self.no_mem_embed.to(backbone_feats.dtype)

        return backbone_feats, high_res_features

    def forward(self, video_frames: List[Tensor], seg_tokens: List[Tensor], seg_meta: Dict[str, Any], resize_to_original_dims: bool, **kwargs):
        """
        video_frames: List of length batch size with tensors each of shape [T, 3, H, W] (T may differ across samples). Should be RGB images normalized to [0, 1]
        seg_tokens: List of length batch size with tensors of shape [M, C] (M = number of objects)

        Returns: List of length batch size with tensors of shape [M, T, H, W]
        """
        if kwargs.get("video_features", None) is not None:
            print(
                f"WARN: sam2 seg head has its own image encoder, but the forward function also received video features from outside. "
                f"These will be ignored and are wasted computation."
            )
        clip_lens = [x.shape[0] for x in video_frames]
        num_objs = [x.shape[0] * self.n_seg_queries for x in seg_tokens]

        video_frames = torch.cat(video_frames)  # [BT, 3, H, W]
        # dtype = video_frames.dtype
        dtype = next(self.image_encoder.parameters()).dtype
        video_frames = video_frames.to(dtype)
        backbone_feats, high_res_feats = self.encode_video_frames(video_frames)

        backbone_feats = list(torch.split(backbone_feats, clip_lens, 0))
        high_res_feats = [torch.split(feats, clip_lens, 0) for feats in high_res_feats]  # outer list over scale, inner list over batch
        high_res_feats = list(zip(*high_res_feats))  # outer list over batch, inner list over scale

        seg_tokens = torch.cat(seg_tokens)
        seg_tokens = self.proj_token(seg_tokens) # [sum(M), Q* C]
        seg_tokens = rearrange(seg_tokens, "M (Q C) -> (M Q) C", Q=self.n_seg_queries)

        sparse_embed, dense_embed = self.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )  # dense_embed: [1, C, H, W]
        assert sparse_embed.numel() == 0
        sparse_embed = seg_tokens.unsqueeze(1)  # [sum(num_objs * Q), 1, C]
        dense_embed = dense_embed.squeeze(0)  # [C, H, W]

        sparse_embed = torch.split(sparse_embed, num_objs, 0)
        image_pe = self.prompt_encoder.get_dense_pe().to(dtype)
        
        pred_masks_all = []
        for sparse_embed_vid, backbone_feats_vid, high_res_feats_vid, meta_vid in zip(
            sparse_embed, backbone_feats, high_res_feats, seg_meta
        ):
            # sparse_embed_vid: [N*Q, 1, C] (N = num objects in current video), Q = num seg queries per object
            # backbone_feats_vid: [T, C, H, W]
            # high_res_feats_per_vid: List, each element = [T, C, H', W']

            dense_embed_vid = repeat(dense_embed, "C H W -> N C H W", N=sparse_embed_vid.shape[0])
            backbone_feats_vid = backbone_feats_vid.unsqueeze(1).to(dtype)
            high_res_feats_vid = [x.unsqueeze(1) for x in high_res_feats_vid]  # each element: [T, 1, C, H', W']

            pred_masks_vid = []
            for t, embed_per_frame in enumerate(backbone_feats_vid):
                high_res_feats_per_frame = [x[t] for x in high_res_feats_vid]

                low_res_masks, iou_predictions, _, _ = self.mask_decoder(
                    image_embeddings=embed_per_frame,
                    image_pe=image_pe,
                    sparse_prompt_embeddings=sparse_embed_vid,
                    dense_prompt_embeddings=dense_embed_vid,
                    multimask_output=False,
                    repeat_image=True,
                    high_res_features=high_res_feats_per_frame
                )  # [N*Q, 1, H, W] (H, W at 1/4 the input resolution), [N*Q, 1]

                pred_masks = self.postprocess_masks(
                    low_res_masks,
                    meta_dict=meta_vid,
                    resize_to_original_dims=resize_to_original_dims
                )  # [N*Q, 1, H', W'] (H' and W' are the original image sizes after reversing resizing and padding)

                pred_masks_vid.append(pred_masks)
            
            pred_masks_vid = torch.cat(pred_masks_vid, 1)  # [N*Q, T, H, W]

            # argmax over per-object queries
            pred_masks_vid = rearrange(pred_masks_vid, "(N Q) T H W -> N Q T H W", Q=self.n_seg_queries)
            pred_masks_vid = pred_masks_vid.max(1).values
            pred_masks_all.append(pred_masks_vid)

        return pred_masks_all

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        meta_dict: Dict[str, Any],
        resize_to_original_dims: bool
    ):
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        dtype = masks.dtype
        img_size = 1024

        masks = F.interpolate(
            masks.float(),
            (img_size, img_size),
            mode="bilinear",
            align_corners=False,
        )

        # if not resize_to_original_dims:
        #     return masks
        
        # first remove padding
        reverse_padding = [-1 * p for p in meta_dict['padding']]
        masks = F.pad(masks, reverse_padding)
        assert list(masks.shape[-2:]) == list(meta_dict["resized_image_size"]), f"Shape mismatch: {masks.shape}, {meta_dict['resized_image_size']}"

        if not resize_to_original_dims:
            return masks

        # then resize to original dims
        tgt_h, tgt_w = meta_dict["orig_image_size"]

        # masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(
            masks, (tgt_h, tgt_w), mode="bilinear", align_corners=False
        )
        return masks
