import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os.path as osp
import multiprocessing as mp
import einops
import io
import random
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import json
from typing import Dict, Sequence
from glob import glob
from PIL import Image

import torch
import torch.nn.functional as F
import transformers
import pycocotools.mask as mt
import warnings

from llava.constants import DEFAULT_VIDEO_TOKEN, DEFAULT_VID_SEG_TOKEN
from torch.utils.data import Dataset

from llava.vision_utils import get_resize_padding_params

from llava.paths import Paths
from llava.train.data_classes import DataArguments
from llava.train.preprocess import preprocess, preprocess_multimodal
from llava.train.vision_augmentation import compute_mask_containing_video_crop
from llava import distributed_utils as dist_utils

try:
    from llava.internal import vicas_dataset_utils
    INTERNAL_IMPORTED = True
except ImportError as _:
    INTERNAL_IMPORTED = False


def expand2square_batch(pil_imgs, background_color):
    # seg_frames: [T, C, H, W] (uint8)
    # seg_masks: [T, N, H, W] (bool)
    result_images = []

    width, height = pil_imgs[0].size
    assert all([tuple(img.size) == (width, height) for img in pil_imgs])

    # if seg_masks is not None:
    #     assert seg_frames.shape[1:3] == seg_masks.shape[-2:]
        # assert (height, width) == tuple(seg_masks.shape[-2:]), f"Mismatch between image ({height, width}) and mask ({tuple(mask.shape[-2:])}) sizes"

    for pil_img in pil_imgs:
        if width == height:
            result = pil_img

        if width > height:
            result = Image.new(
                pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))

        else:
            result = Image.new(
                pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))

        result_images.append(result)

    return result_images


class ViCaSDataset(Dataset):
    prompt_list = [
        "Describe what is happening in the video in detail. Talk about the events, the main objects/actors and also briefly describe the background scene.",
        "Provide a detailed description of the video, focusing on the events, key participants or objects, and the background setting.",
        "Explain in detail what is occurring in the video, including the main actions, important figures or items, and the surrounding environment.",
        "Give a thorough account of the video, describing the events, the central actors or objects, and the scene in the background",
        "Describe the video comprehensively, covering the actions taking place, the primary subjects or objects involved, and the background setting."
    ]


    def __init__(self, 
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super().__init__()

        self.samples = []

        self.internal_dataset = self.internal_frames_dataset = self.video_dir = self.video_frames_dir = None

        if data_args.use_falcon_dataset:
            vicas_dataset_utils.create_dataset_loader(self, data_args)
        else:
            self.video_dir = Paths.vicas_videos_dir()
            self.video_frames_dir = Paths.vicas_video_frames_dir()

        # load train split video IDs
        with open(Paths.vicas_split_json(version=data_args.vicas_version, split="train"), 'r') as fh:
            train_video_ids = json.load(fh)

        # gather sample list
        json_files = sorted(glob(osp.join(Paths.vicas_annotations_dir(data_args.vicas_version), "*.json")))
        samples_caption = []
        samples_referral = []
        assert not (data_args.exclude_captions and data_args.exclude_seg) # sanity check

        for f in json_files:
            with open(f, 'r') as fh:
                content = json.load(fh)

            if content['video_id'] not in train_video_ids:
                continue

            if not data_args.exclude_captions:
                num_reworded_captions = len(content["reworded_en_captions"]) # len(content.get("reworded_en_captions", []))
                samples_caption.append((f, 'caption', -1))  # -1 denotes original caption
                samples_caption.extend([(f, 'caption', j) for j in range(num_reworded_captions)])

            if not data_args.exclude_seg:
                for i in range(len(content['object_referrals'])):
                    samples_referral.append((f, 'mask', i))

        if len(samples_referral) < len(samples_caption) and not data_args.exclude_seg:
            n_pad = len(samples_caption) - len(samples_referral)
            pad_samples = random.choices(samples_referral, k=n_pad)
            samples_referral.extend(pad_samples)

        self.samples = samples_caption + samples_referral

        dist_utils.print_once(f"[ViCaS] Video Caption samples: {len(samples_caption)}")
        dist_utils.print_once(f"[ViCaS] LG-VIS samples: {len(samples_referral)}")
        dist_utils.print_once(f"[ViCaS] Total samples: {len(self.samples)}")

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.n_epochs = 1

    @property
    def is_falcon_dataset(self):
        return self.data_args.use_falcon_dataset

    def set_num_epochs(self, n: int):
        self.n_epochs = n

    def init_internal_dataset_readers(self, local_pid, local_size):
        self.internal_dataset.init_reader(local_pid, local_size)
        if self.internal_frames_dataset is not None:
            self.internal_frames_dataset.init_reader(local_pid, local_size)

    def filter_indices(self, indices_to_keep: List[int]):
        self.samples = [self.samples[i] for i in indices_to_keep]

    def __len__(self):
        return int(len(self.samples) * self.n_epochs)

    @property
    def modality_lengths(self):
        return [1 for _ in range(len(self))] 

    def get_video(self, json_content):
        if self.is_falcon_dataset:
            video_bytes = vicas_dataset_utils.get_video_bytes(self, json_content)
        else:
            filepath = osp.join(self.video_dir, json_content['filename'])
            assert osp.exists(filepath), f"Video not found: {filepath}"
            with open(filepath, 'rb') as fh:
                video_bytes = fh.read()

        return io.BytesIO(video_bytes)

    def get_seg_frames_and_masks(self, json_content, track_ids: Optional[List[int]] = None):
        frames = []
        masks_seq = []
        filenames = []
        frame_indices = []
        meta = {}

        for t, segs_t in enumerate(json_content["segmentations"]):
            if not segs_t["is_gt"]:
                continue

            if self.internal_frames_dataset is None:
                image_path = osp.join(self.video_frames_dir, f"{json_content['video_id']:06d}", segs_t['filename'])
                assert osp.exists(image_path), f"Video frame not found: {image_path}"
                with open(image_path, 'rb') as fh:
                    image_bytes = fh.read()

            else:
                image_bytes = vicas_dataset_utils.get_image_bytes(self, json_content, segs_t)

            filenames.append(segs_t['filename'])
            frame_indices.append(t)

            image_bytes = io.BytesIO(image_bytes)
            image = np.array(Image.open(image_bytes))
            h, w = image.shape[:2]
            meta['orig_image_size'] = (h, w)
            frames.append(image)

            if track_ids is not None:
                masks_t = [np.zeros((h, w), np.uint8) for _ in range(len(track_ids))]

                for track_id, mask_rle in zip(segs_t["track_ids"], segs_t["mask_rles"]):
                    if track_id not in track_ids:
                        continue

                    idx = track_ids.index(track_id)
                    mask_rle["counts"] = mask_rle["counts"].encode("utf-8")
                    masks_t[idx] = mt.decode(mask_rle).astype(np.uint8)

                masks_t = torch.from_numpy(np.stack(masks_t, 0))  # [N, H, W]
                masks_seq.append(masks_t)

        if track_ids is not None:
            masks_seq = torch.stack(masks_seq, 1).bool()  # [N, T, H, W]

        frames = torch.from_numpy(np.stack(frames))  # [T, H, W, 3], RGB, [0-255]
        frames = einops.rearrange(frames, "T H W C -> T C H W")
        frame_indices = torch.tensor(frame_indices, dtype=torch.float32) / float(len(json_content["segmentations"]) - 1)

        if frames.shape[0] > self.data_args.max_seg_frames:
            keep_indices = torch.linspace(0, frames.shape[0]-1, self.data_args.max_seg_frames, dtype=torch.long).round().to(device=frames.device)
            frames = frames[keep_indices]
            frame_indices = frame_indices[keep_indices]
            
            filenames = [filenames[i] for i in keep_indices.tolist()]
            if track_ids is not None:
                masks_seq = masks_seq[:, keep_indices]

        meta.update({
            'video_id': json_content['video_id'],
            'filenames': filenames,
            "timestamps": frame_indices,
            'track_ids': track_ids
        })

        if track_ids is None:
            return frames, None, meta
        else:
            return frames, masks_seq, meta

    def __getitem__(self, index):
        for n in range(3):
            try:
                sample = self.parse_sample(index)
            except Exception as exc:
                index = random.randint(0, len(self)-1)
                continue

            return sample
                
        raise RuntimeError(f"Failed to parse sample after 3 tries")

    def parse_sample(self, index):
        index = index % len(self.samples)
        try:
            json_path, sample_type, sub_index = self.samples[index]
            # sub_index = 0
            with open(json_path, 'r') as fh:
                content = json.load(fh)

            video_obj = self.get_video(content)

            video_id = content["video_id"]
            caption = content["caption_parsed_en_gpt"]  # "caption_parsed_en"

            if sample_type == 'caption':
                prompt = random.choice(self.prompt_list)
                conversation = [
                    {
                        "from": "human",
                        "value": DEFAULT_VIDEO_TOKEN + ("\n" + prompt if self.data_args.use_text_prompt else "")
                    },
                    {
                        "from": "gpt",
                        "value": caption
                    }
                ]
                seg_frames, seg_masks, seg_meta = self.get_seg_frames_and_masks(content, None)  # seg_frames: [T, C, H, W], seg_masks: None

            elif sample_type == 'mask':
                question = content["object_referrals"][sub_index]["prompt"]
                track_ids = content["object_referrals"][sub_index]["track_ids"]

                conversation = [
                    {
                        "from": "human",
                        "value": f"{DEFAULT_VIDEO_TOKEN}\n{question} Please output the segmentation mask." 
                    },
                    {
                        "from": "gpt",
                        "value": DEFAULT_VID_SEG_TOKEN * len(track_ids)
                    }
                ]

                seg_frames, seg_masks, seg_meta = self.get_seg_frames_and_masks(content, track_ids)  # seg_frames: [T, C, H, W], seg_masks: [N, T, H, W]

            else:
                raise ValueError(f"Unexpected sample type: {sample_type}")

            # ======================================================================================================
            video_processor = self.data_args.video_processor
            video, timestamps = video_processor.load_video(video_obj, self.data_args.num_frames, return_normalized_timestamps=True)  # list of PIL images
            seg_meta["llm_timestamps"] = timestamps

            assert self.data_args.image_aspect_ratio == 'pad', f"Unsupported image aspect ratio: {self.data_args.image_aspect_ratio}"

            image_mean = tuple(int(x * 255) for x in video_processor.image_mean)
            video = expand2square_batch(video, image_mean)
            image_size = video[0].size
            video = video_processor.preprocess(video, return_tensors='pt')  # [T, C, H, W]

            if seg_masks is not None:
                seg_frames, seg_masks, seg_meta = self.augment_sample(seg_frames, seg_masks, seg_meta, crop_factor=0.7)
            seg_frames, seg_masks, seg_meta = self.preprocess_seg_inputs(seg_frames, seg_masks, seg_meta, process_masks=True)

            sources = preprocess_multimodal([conversation], self.data_args)
            data_dict = preprocess(sources, self.tokenizer, has_image=True)
            seg_meta['sub_index'] = sub_index
            # ==========================================================================================================

            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])
            
            data_dict['image'] = video
            data_dict['image_size'] = image_size

            if not self.data_args.exclude_seg:
                data_dict['seg_frames'] = seg_frames
                data_dict['seg_masks'] = seg_masks
                data_dict['seg_meta'] = seg_meta

            return data_dict

        except Exception as e:
            print(f'Error processing JSON at {json_path} (sub-index {sub_index}): {e}')
            raise e

    def preprocess_seg_inputs(self, seg_frames, seg_masks, seg_meta, process_masks=False):
        # seg_frames: [T, C, H, W] (uint8), 0-255
        # seg_masks: [T, N, H, W] or None
        seg_frames = seg_frames.to(torch.float32)

        # if self.data_args.video_processor.do_rescale:
        seg_frames = seg_frames / 255.

        if self.data_args.video_processor.do_normalize and self.data_args.seg_frames_shared_encoder:
            img_mean = torch.tensor(self.data_args.video_processor.image_mean, dtype=torch.float32)[None, :, None, None]
            img_std = torch.tensor(self.data_args.video_processor.image_std, dtype=torch.float32)[None, :, None, None]
            seg_frames = (seg_frames - img_mean) / img_std

        if seg_masks is not None:
            assert seg_frames.shape[-2:] == seg_masks.shape[-2:]

        # resize to larger dim == target size, and then center pad the shorter dim
        tgt_size = self.data_args.seg_image_size
        h, w = seg_frames.shape[-2:]

        (h, w), (pad_left, pad_right, pad_top, pad_bottom) = get_resize_padding_params(h, w, tgt_size, pad_mode=self.data_args.seg_pad_mode)
        # pad_left = pad_right = pad_top = pad_bottom = 0

        seg_frames = F.interpolate(seg_frames, (h, w), mode='bilinear', align_corners=False)
        seg_frames = F.pad(seg_frames, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

        if seg_masks is not None and process_masks:
            dtype = seg_masks.dtype  # F.interpolate does not work with bool dtype
            assert dtype in (torch.uint8, torch.bool)
            seg_masks = seg_masks.float()
            seg_masks = F.interpolate(seg_masks, (h, w), mode='bilinear', align_corners=False) > 0.5
            # seg_masks = F.interpolate(seg_masks, (h, w), mode='nearest-exact')
            # seg_masks = F.pad(seg_masks, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
            seg_masks = seg_masks.to(dtype)

        seg_meta["resized_image_size"] = (h, w)
        seg_meta["padding"] = (pad_left, pad_right, pad_top, pad_bottom)

        return seg_frames, seg_masks, seg_meta

    def augment_sample(self, seg_frames, seg_masks, seg_meta, crop_factor=0.75):
        # seg_frames: [T, C, H, W] (uint8), 0-255
        # seg_masks: [N, T, H, W] or None
        height, width = seg_frames.shape[-2:]
        crop_height = int(height * crop_factor)
        crop_width = int(width * crop_factor)
        
        merged_masks = torch.any(seg_masks, 0)  # [T, H, W]
        try:
            ret = compute_mask_containing_video_crop(merged_masks, (crop_height, crop_width))
        except Exception as exc:
            print(f"Error in dataset augmentation: {exc}")
            ret = None

        if ret is None:
            seg_meta.update({
                "crop_topleft": (0, 0),
                "orig_image_size": (height, width),
                "precrop_image_size": (height, width)
            })
        else:          
            x1, y1 = ret
            seg_frames = seg_frames[:, :, y1:y1+crop_height, x1:x1+crop_width]
            seg_masks = seg_masks[:, :, y1:y1+crop_height, x1:x1+crop_width]
            seg_meta["precrop_image_size"] = seg_meta.pop("orig_image_size")
            seg_meta.update({
                "crop_topleft": (y1, x1),
                "orig_image_size": (crop_height, crop_width),
            })

        return seg_frames, seg_masks, seg_meta
