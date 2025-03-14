#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod
from einops import rearrange
from typing import List, Dict, Any, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector
from .seg_head.builder import build_segmentation_head

from llava.constants import (
    IGNORE_INDEX, 
    IMAGE_TOKEN_INDEX, 
    DEFAULT_IMAGE_PATCH_TOKEN, 
    DEFAULT_IM_START_TOKEN, 
    DEFAULT_IM_END_TOKEN, 
    DEFAULT_VID_START_TOKEN,
    DEFAULT_VID_END_TOKEN,
    DEFAULT_SF_VID_SEPARATOR_TOKEN,
    DEFAULT_VID_SEG_TOKEN
)

from llava import distributed_utils as dist_utils


class LlavaMetaModel:
    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

        if getattr(config, "seg_head", None) not in (None, ""):
            self.seg_head = build_segmentation_head(config)
        else:
            self.seg_head = None

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_segmentation_head(self):
        return getattr(self, 'seg_head', None)

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.seg_head = getattr(model_args, 'seg_head', None)
        self.config.seg_num_queries = getattr(model_args, "seg_num_queries", None)
        self.config.seg_backbone = getattr(model_args, "seg_backbone", None)
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if self.get_segmentation_head() is None and model_args.seg_head:
            self.seg_head = build_segmentation_head(self.config)

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def avgpool_video_features(self, video_features):
        im_dim = math.sqrt(video_features.shape[2])
        assert int(im_dim) == im_dim, f"Expected square image, but got video_features={video_features.shape}"
        im_dim = int(im_dim)
        bs, clip_len = video_features.shape[:2]
        video_features = rearrange(video_features, "B T (H W) C -> (B T) C H W", H=im_dim, W=im_dim)
        video_features = F.adaptive_avg_pool2d(video_features, (4, 4))
        video_features = rearrange(video_features, "(B T) C H W -> B T (H W) C", B=bs, T=clip_len)
        return video_features

    def encode_video(self, video):
        # seg_frames: List[[T, C, H, W]]
        video_features = self.get_model().get_vision_tower()(video)  # [B, T, N, C]
        video_features = self.get_model().mm_projector(video_features)

        # split into slow and fast features
        num_slow_frames = self.config.num_slow_frames
        if num_slow_frames != self.config.num_frames:
            frame_ids = torch.linspace(0, video_features.shape[1]-1, num_slow_frames, dtype=torch.int64, device=video_features.device)
            video_features_slow = torch.index_select(video_features, 1, frame_ids)
            video_features_fast = self.avgpool_video_features(video_features)
        else:
            video_features_fast = None
            video_features_slow = video_features

        return video_features_slow, video_features_fast

    def encode_seg_frames(self, seg_frames: List[torch.Tensor]):
        if self.get_model().seg_head.has_image_encoder:
            return None
        
        clip_lens = [x.shape[0] for x in seg_frames]
        seg_frames = torch.cat(seg_frames, 0)  # [BT, C, H, W]
        # print(seg_frames.shape)
        seg_frames_features = self.get_model().get_vision_tower()(seg_frames)  # [BT, N, C]
        return torch.split(seg_frames_features, clip_lens, 0)

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None, seg_frames=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, None # <-- seg_frame_features, video_features_fast

        # if getattr(self.config, "video_mode", False):
        assert self.config.video_mode
        
        # images: list (batch size). Each element: [T, C, H, W]
        if isinstance(images, (list, tuple)):
            video = torch.stack(images)  # [B, T, C, H, W]
        else:
            video = images # lol

        video_features_slow, video_features_fast = self.encode_video(video)  # [B, T, N, C], [B, T', N, C]
        seg_frame_features = self.encode_seg_frames(seg_frames) if seg_frames is not None else None  # List[[T, N, C]] or None if seg_head has its own encoder

        if video_features_fast is None:
            image_features = rearrange(video_features_slow, "B T N C -> (B T) N C")
        else:
            image_features = []
            for slow_b, fast_b in zip(video_features_slow, video_features_fast):
                image_features.extend(slow_b.unbind(0))
                image_features.extend(fast_b.unbind(0))

        # image_features = rearrange(video_features, "B T N C -> (B T) N C")
        # print("image_features: ", image_features.shape)

        # elif type(images) is list or images.ndim == 5:  # image-mode. Not used by us.
        #     if type(images) is list:
        #         images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
        #     concat_images = torch.cat([image for image in images], dim=0)
        #     image_features = self.encode_images(concat_images)  # concat_images: [36, 3, 336, 336], image_features: [36, 576, 4096]. 576 = 24 * 24 = (336 / 14) * (336 / 14)
        #     split_sizes = [image.shape[0] for image in images]
        #     image_features = torch.split(image_features, split_sizes, dim=0)
        #     mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')  # spatial_unpad
        #     image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')  # anyres

        #     if mm_patch_merge_type == 'flat':
        #         image_features = [x.flatten(0, 1) for x in image_features]
        #     elif mm_patch_merge_type.startswith('spatial'):
        #         new_image_features = []
        #         for image_idx, image_feature in enumerate(image_features):
        #             if image_feature.shape[0] > 1:
        #                 base_image_feature = image_feature[0]  
        #                 image_feature = image_feature[1:]
        #                 height = width = self.get_vision_tower().num_patches_per_side
        #                 assert height * width == base_image_feature.shape[0]
        #                 if image_aspect_ratio == 'anyres':
        #                     # image_sizes[image_width] = [612, 612] 
        #                     num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
        #                     # num_patch_width, num_patch_height = 2, 2
        #                     image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
        #                 else:
        #                     raise NotImplementedError
        #                 if 'unpad' in mm_patch_merge_type:
        #                     image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()  # [4096, 2, 24, 2, 24] (2's come from num_patch_width, num_patch_height)
        #                     image_feature = image_feature.flatten(1, 2).flatten(2, 3)  # [4096, 48, 48]
        #                     image_feature = unpad_image(image_feature, image_sizes[image_idx])  # [4096, 48, 48] (because there was no padding on the orig image)
        #                     # self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1) shape: [4096, 48, 1] i.e. add a 'padding' feature at the end of every row
        #                     image_feature = torch.cat((
        #                         image_feature,
        #                         self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
        #                     ), dim=-1)  # [4096, 48, 49] (width dim increased from 48 to 49)
        #                     image_feature = image_feature.flatten(1, 2).transpose(0, 1)  # [2352, 4096]. 2352 = 48 * 49
        #                 else:
        #                     image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
        #                     image_feature = image_feature.flatten(0, 3)
        #                 image_feature = torch.cat((base_image_feature, image_feature), dim=0)
        #             else:
        #                 image_feature = image_feature[0]
        #                 if 'unpad' in mm_patch_merge_type:
        #                     image_feature = torch.cat((
        #                         image_feature,
        #                         self.model.image_newline[None].to(image_feature.device)
        #                     ), dim=0)
        #             new_image_features.append(image_feature)
        #         image_features = new_image_features  # list (batch size). Each element = [N, C]
        #     else:
        #         raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")

        # else:
        #     image_features = self.encode_images(images)

        # At this point, image_features is a list (one element per sample) containing tensors of shape [N, C] (N = #tokens per image)

        # TODO: image start / end is not implemented here to support pretraining.
        # if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
        #     raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0  # this is a flat index across all batch samples.
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                print(f"WARN: No image tokens found in input_ids.")
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            assert num_images == (len(image_features) // len(input_ids)), f"num_images={num_images}, len(image_features)={len(image_features)}, batch_size={len(input_ids)}"

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            max_input_len = max([x.shape[0] for x in new_input_embeds])
            if max_input_len > tokenizer_model_max_length:
                print(f"WARN: Input sequence ({max_input_len}) is longer than max sequence length ({tokenizer_model_max_length}) and will be truncated")
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        # print("Max length: ", max_len)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, seg_frame_features, video_features_fast

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_sf_vid_separator_token:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_SF_VID_SEPARATOR_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN], special_tokens=True
            )
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                dist_utils.print_once(f"Restoring input/output embeddings from {model_args.pretrain_mm_mlp_adapter}")
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens >= 4
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

        if model_args.seg_head:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_VID_SEG_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg
