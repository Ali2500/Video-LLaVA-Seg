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


from typing import List, Optional, Tuple, Union, Dict, Any

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

from ..seg_head.mask_loss import MaskLoss


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mask_loss = MaskLoss()

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        seg_masks: Optional[torch.Tensor] = None,
        seg_frames: Optional[torch.Tensor] = None,
        seg_meta: Optional[Dict[str, Any]] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # input_ids_copy = input_ids.clone()

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                seg_frame_features,
                video_features_fast
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
                seg_frames,
            )
        else:
            seg_frame_features = None  # inference mode. Forward pass through seg_head will be done in `generate()`

        if self.model.seg_head is not None:
            output_hidden_states = True

        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        if seg_frames is not None:
            assert labels is not None  # train mode
            output_dict = self.forward_seg_head_train(
                llm_output=output, 
                seg_frames=seg_frames,
                seg_frame_features=seg_frame_features, 
                seg_meta=seg_meta, 
                labels=labels,
                seg_masks=seg_masks,
                video_features_fast=video_features_fast
            )
        else:
            output_dict = output

        return output_dict

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        seg_frames: Optional[torch.Tensor] = None,
        seg_meta: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        # inputs_copy = inputs.clone()
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                seg_frame_features,
                video_features_fast
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes,
                seg_frames=seg_frames
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
            seg_frame_features = None

        output = super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            return_dict_in_generate=True,
            **kwargs
        )

        if seg_frames is not None:
            # concatenate all output embeddings
            output_embeds = torch.cat([x[-1] for x in output.hidden_states], 1)
            # if there were N input embeddings, then the 'new' output of the LLM begins at N-1
            output_embeds = output_embeds[:, inputs_embeds.shape[1]-1:]
            output_ids = output.sequences[:, 1:]  # ignore <start_of_text> token at position 0
            assert output_embeds.shape[:2] == output_ids.shape, f"Shape mismatch: {output_embeds.shape}, {output_ids.shape}"

            output = dict(output)
            output['pred_mask_logits'] = self.forward_seg_head_inference(
                output_ids=output_ids,
                output_embeds=output_embeds,
                seg_frames=seg_frames,
                seg_frame_features=seg_frame_features,
                video_features_fast=video_features_fast,
                seg_meta=seg_meta
            )
        else:
            output = dict(output)

        return output

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

    def forward_seg_head_train(
            self, 
            llm_output: CausalLMOutputWithPast,
            seg_frames,
            seg_frame_features,
            seg_meta,
            labels,
            seg_masks,
            video_features_fast
        ):
        output = llm_output
        hidden_states = output.hidden_states[-1]

        seg_token_batch_idx, seg_token_pos_idx = (labels == self.config.seg_token_idx).nonzero(as_tuple=True)
        seg_token_pos_idx -= 1  # index shift to align labels with logits

        batch_sz = len(seg_masks)
        seg_tokens = []
        num_object_tracks = 0

        for b in range(batch_sz):
            seg_token_pos_idx_b = seg_token_pos_idx[seg_token_batch_idx == b]
            if seg_token_pos_idx_b.numel() == 0:  # captioning sample. seg_masks should be None
                assert seg_masks[b] is None, f"No seg tokens found in sample {b}, but seg_masks exist of shape {seg_masks[b].shape}. Meta info: {seg_meta}"
                continue

            assert torch.is_tensor(seg_masks[b]), f"Seg token exists in labels at {seg_token_pos_idx_b.tolist()}, but no seg_masks are given. Meta info: {seg_meta}"
            assert seg_masks[b].shape[0] == seg_token_pos_idx_b.numel(), f"Shape mismatch: {seg_masks[b].shape}, {seg_token_pos_idx_b}"

            seg_tokens.append(hidden_states[b, seg_token_pos_idx_b])  # [num_objs, C]
            num_object_tracks += seg_token_pos_idx_b.numel()
            
        if seg_tokens:
            dummy_forward_pass = False
        else:
            # none of the batch samples have GT masks (i.e. they are all captioning examples). In this case we make a dummy seg token just to
            # keep the forward pass the same across all GPUs. The mask output for this dummy seg token should be multiplied with zero to avoid
            # messing up the optimization
            seg_tokens = [hidden_states[0, -1].unsqueeze(0)]
            t = seg_frames[0].shape[0]
            # h, w = seg_meta[b]["orig_image_size"]
            # h = w = self.model.config.seg_image_size
            h, w = seg_meta[0]["resized_image_size"]
            seg_masks = [torch.zeros(1, t, h, w, dtype=torch.bool, device=seg_frames[0].device)]
            dummy_forward_pass = True

        try:
            pred_masks = self.model.seg_head(
                video_frames=seg_frames,
                video_features=seg_frame_features,
                seg_tokens=seg_tokens,
                seg_meta=seg_meta,
                video_features_fast=video_features_fast,
                resize_to_original_dims=False
            )  # list of length batch size. Each element is a tensor of shape [N, T, H, W] (N = num objects = num seg tokens)

        except Exception as exc:
            print(f"Error occurred during seg_head forward pass for sample with meta info: {seg_meta}")
            raise exc

        assert len(pred_masks) == batch_sz
        try:
            loss_mask_dict = self.mask_loss(
                pred_masks=pred_masks,
                gt_masks=seg_masks,
                dummy_forward_pass=dummy_forward_pass
            )
        except Exception as exc:
            print(f"Error occurred during mask loss computation for sample with meta info: {seg_meta}")
            raise exc

        loss_text = output["loss"]

        output_dict = {
            "loss": loss_text + loss_mask_dict["loss_mask"],
            "loss_text": loss_text,
            "loss_mask": loss_mask_dict["loss_mask"],
            "loss_mask_dice": loss_mask_dict["loss_mask_dice"],
            "loss_mask_ce": loss_mask_dict["loss_mask_ce"],
            "mask_ious": loss_mask_dict["mask_ious"],
            "num_object_tracks": torch.tensor(num_object_tracks, dtype=torch.float32, device=loss_text.device)
        }

        return output_dict

    def forward_seg_head_inference(
            self, 
            output_ids: torch.Tensor,
            output_embeds: torch.Tensor, 
            seg_frames: List[torch.Tensor],
            seg_frame_features: List[torch.Tensor],
            video_features_fast: torch.Tensor,
            seg_meta: List[Dict[str, Any]]
        ):
        assert len(seg_meta) == output_ids.shape[0] == output_embeds.shape[0] == 1  # batch size = 1
        
        seg_token_pos_idx = (output_ids.squeeze(0) == self.config.seg_token_idx).nonzero(as_tuple=False).squeeze(1)
        if seg_token_pos_idx.numel() == 0:
            print(f"WARN: No seg token found in output. Assuming that token at index 0 with ID {output_ids[0].item()} is the seg token")
            seg_token_pos_idx = torch.tensor([0]).to(seg_token_pos_idx)

        seg_tokens = output_embeds[0, seg_token_pos_idx]  # [num_objs, C]
        try:
            pred_masks = self.model.seg_head(
                video_frames=seg_frames,
                video_features=seg_frame_features,
                seg_tokens=[seg_tokens],
                video_features_fast=video_features_fast,
                seg_meta=seg_meta,
                resize_to_original_dims=True
            )  # List of length 1 (batch size). Tensor of shape [num_objs, T, H, W] (N = num objects = num seg tokens)

            assert len(pred_masks) == 1
            pred_mask_logits = pred_masks[0]

        except Exception as exc:
            print(f"Error occurred during seg_head forward pass for sample with meta info: {seg_meta[0]}")
            raise exc

        return pred_mask_logits


AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
