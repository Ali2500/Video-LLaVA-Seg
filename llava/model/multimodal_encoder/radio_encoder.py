import torch
import torch.nn as nn

from transformers import AutoModel, AutoConfig
from llava.model.multimodal_encoder.clip_video_processor import CLIPVideoProcessor, CLIPImageProcessor


class RADIOVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.train_vision_tower = getattr(args, 'unfreeze_mm_vision_tower', False)
        self._model_args = args # this will actually be the LlavaConfig object at inference

        if not delay_load:
            self.load_model()
        elif self.train_vision_tower:
            self.load_model()
        else:
            self.cfg_only = AutoConfig.from_pretrained(self.vision_tower_name, trust_remote_code=True)
    
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        for arg in args:
            if isinstance(arg, torch.dtype):
                self.vision_tower.radio_model.input_conditioner.dtype = arg
                break
                
        if 'dtype' in kwargs:
            self.vision_tower.radio_model.input_conditioner.dtype = kwargs['dtype']

        return self

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = self.adjust_processor(CLIPImageProcessor.from_pretrained(self.vision_tower_name))
        self.video_processor = self.adjust_processor(CLIPVideoProcessor.from_pretrained(self.vision_tower_name))

        # for proc in (self.image_processor, self.video_processor):
        #     proc.size['shortest_edge'] = self._model_args.image_size
        #     proc.crop_size['height'] = proc.crop_size['width'] = self._model_args.image_size

        self.vision_tower = AutoModel.from_pretrained(
            self.vision_tower_name, 
            device_map=device_map, 
            trust_remote_code=True, 
            code_revision='b56a0021dfa3125edc9ec0831b572f0ef72f7a9f'
        )
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True
        self.vision_tower.radio_model.input_conditioner.dtype = self.dtype

    def adjust_processor(self, processor):
        assert self._model_args.image_size is not None
        processor.size['shortest_edge'] = self._model_args.image_size
        processor.crop_size['height'] = processor.crop_size['width'] = self._model_args.image_size
        processor.do_resize = processor.do_center_crop = True
        return processor

    def forward(self, images):
        with torch.set_grad_enabled(self.train_vision_tower):
            if type(images) is list:
                image_features = []
                for image in images:
                    image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0))
                    image_feature = image_forward_out.to(image.dtype)
                    image_features.append(image_feature)
            else:
                if images.ndim == 5:  # video input
                    # images: [B, T, C, H, W]
                    bs, clip_len = images.shape[:2]
                    images = images.flatten(0, 1)
                    _, image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype))
                    image_features = image_forward_outs.to(images.dtype)
                    image_features = image_features.reshape(bs, clip_len, *image_features.shape[1:])
                else:
                    # images: [B, C, H, W]
                    _, image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype))
                    image_features = image_forward_outs.to(images.dtype)

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
        if self.vision_tower_name == 'nvidia/RADIO':
            return 1280
        elif self.vision_tower_name == 'nvidia/RADIO-L':
            return 1024
        elif self.vision_tower_name == 'nvidia/RADIO-B':
            return 768
        elif self.vision_tower_name == 'nvidia/E-RADIO':
            return 1536
        else:
            raise ValueError(f"Invalid model name: {self.vision_tower_name}")

    @property
    def num_patches_per_side(self):
        return self._model_args.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self._model_args.image_size // self.config.patch_size) ** 2
