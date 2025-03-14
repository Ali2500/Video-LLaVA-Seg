import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

from llava.model.multimodal_encoder.clip_video_processor import CLIPVideoProcessor


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.train_vision_tower = getattr(args, 'unfreeze_mm_vision_tower', False)
        self.image_size = args.image_size
        self.num_frames = args.num_frames
        self.num_slow_frames = args.num_slow_frames

        if not delay_load:
            self.load_model()
        elif self.train_vision_tower:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.video_processor = CLIPVideoProcessor.from_pretrained(self.vision_tower_name)
        
        for proc in (self.image_processor, self.video_processor):
            proc.size['shortest_edge'] = self.image_size
            proc.crop_size['height'] = proc.crop_size['width'] = self.image_size

        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)
        self.resize_position_embedding()

        self.is_loaded = True

    @torch.no_grad()
    def resize_position_embedding(self):
        default_image_size = self.vision_tower.vision_model.embeddings.image_size
        if self.image_size == default_image_size:
            return
        
        patch_size = self.vision_tower.vision_model.embeddings.patch_size
        default_spatial_dim_size = default_image_size // patch_size
        assert self.image_size % patch_size == 0, f"Image size ({self.image_size}) must be divisible by patch size ({patch_size})"

        pos_embedding = self.vision_tower.vision_model.embeddings.position_embedding.weight # [577, 1024]. First feature is cls token
        cls_embed = pos_embedding[:1]
        spatial_embed = rearrange(pos_embedding[1:], "(H W) C -> 1 C H W", H=default_spatial_dim_size, W=default_spatial_dim_size)

        new_spatial_dim_size = self.image_size // patch_size
        spatial_embed = F.interpolate(spatial_embed, (new_spatial_dim_size, new_spatial_dim_size), mode='bicubic', align_corners=True)
        spatial_embed = rearrange(spatial_embed.squeeze(0), "C H W -> (H W) C")

        new_pos_embedding = nn.Embedding(spatial_embed.shape[0] + 1, spatial_embed.shape[1])
        new_pos_embedding.weight.copy_(torch.cat((cls_embed, spatial_embed), 0))
        self.vision_tower.vision_model.embeddings.position_embedding = new_pos_embedding

        # update position_ids buffer
        new_pos_ids = torch.arange(new_pos_embedding.num_embeddings).to(self.vision_tower.vision_model.embeddings._buffers['position_ids'])
        self.vision_tower.vision_model.embeddings._buffers['position_ids'] = new_pos_ids

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def forward(self, images):
        with torch.set_grad_enabled(self.train_vision_tower):
            if type(images) is list:
                image_features = []
                for image in images:
                    image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                    image_feature = self.feature_select(image_forward_out).to(image.dtype)
                    image_features.append(image_feature)
            else:
                if images.ndim == 5:  # video input
                    # images: [B, T, C, H, W]
                    bs, clip_len = images.shape[:2]
                    images = images.flatten(0, 1)
                    image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
                    image_features = self.feature_select(image_forward_outs).to(images.dtype)
                    image_features = image_features.reshape(bs, clip_len, *image_features.shape[1:])
                else:
                    # images: [B, C, H, W]
                    image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
                    image_features = self.feature_select(image_forward_outs).to(images.dtype)

            return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2



class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
