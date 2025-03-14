import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from collections import defaultdict
import os
import os.path as osp
import pickle
import multiprocessing as mp
import copy
import io
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import json
from typing import Dict, Sequence
from glob import glob
from tqdm import tqdm
from PIL import Image

import torch
import transformers

from llava.constants import DEFAULT_VIDEO_TOKEN
from torch.utils.data import Dataset

from llava.paths import Paths
from llava.train.data_classes import DataArguments
from llava.train.preprocess import preprocess, preprocess_multimodal
from llava import distributed_utils as dist_utils

try:
    from llava.internal import video_caption_dataset_utils
    INTERNAL_IMPORTED = True
except ImportError as _:
    INTERNAL_IMPORTED = False


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(
            pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(
            pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


class VideoCaptionDataset(Dataset):
    prompt_list = [
        "Describe what is happening in the video in a few sentences.",
        "Summarize the events occurring in the video in a few sentences.",
        "Provide a brief description of the actions taking place in this video.",
        "Explain what is happening in the video scene by scene.",
        "Describe the main activities shown in this video clip.",
        "Give an overview of the key events happening in the video.",
        "Write a short narrative of the events depicted in the video.",
        "Detail the sequence of actions occurring in this video.",
        "Capture the essence of the video by describing the actions shown.",
        "Narrate the key moments of the video in a few lines.",
        "Please faithfully summarize the video in a few sentences"
    ]

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super().__init__()

        self.internal_format_datasets = {}
        self.internal_format_sample_indices = []

        if data_args.use_falcon_dataset:
            assert INTERNAL_IMPORTED
            video_caption_dataset_utils.create_dataset_loader(self, data_args)

        else:
            self.json_paths = []
            self.dataset_paths = {
                "WebVid10M": Paths.webvid_train_dir(),
                "Panda70M": Paths.panda70m_train_dir()
            }

            if data_args.data_version == "all_available":
                dist_utils.print_once(f"Recursing over data directories to find training samples. This may take a while...")
                for name, path in self.dataset_paths.items():
                    # get all shard directories
                    shard_dirs = sorted([x for x in os.listdir(path) if osp.isdir(osp.join(path, x))])
                    assert shard_dirs, f"No shard directories found under {path}"
                    json_paths = []

                    for d in shard_dirs:
                        json_paths.extend([(name, osp.join(d, x)) for x in sorted(os.listdir(osp.join(path, d))) if x.endswith(".json")])

                    assert len(json_paths) > 0, f"No JSON files found in {path}"
                    self.json_paths.extend(json_paths)

            else:
                self.json_paths = self.get_data_version_paths(data_args.data_version) # content['json_paths']

            num_samples = defaultdict(lambda: 0)
            for ds_name, _ in self.json_paths:
                num_samples[ds_name] += 1
            dist_utils.print_once(f"Caption dataset video count: {dict(num_samples)}")

        self.tokenizer = tokenizer
        self.data_args = data_args

    @property
    def is_falcon_dataset(self):
        return self.data_args.use_falcon_dataset

    def init_internal_dataset_readers(self, local_pid, local_size):
        assert INTERNAL_IMPORTED
        for ds in self.internal_format_datasets.values():
            ds.init_reader(local_pid, local_size)

    def get_data_version_paths(self, data_version):
        data_versions_dir = osp.join(osp.dirname(osp.dirname(osp.dirname(__file__))), "llava", "internal", "data_versions")
        assert osp.exists(data_versions_dir), f"Data versions directory not found at {data_versions_dir}"
        pkl_path = osp.join(data_versions_dir, f"{data_version}.pkl")
        dist_utils.print_once(f"Loading video caption dataset file list from {pkl_path}")

        with open(pkl_path, 'rb') as fh:
            content = pickle.load(fh)

        return content['json_paths']

    def filter_indices(self, indices_to_keep: List[int]):
        if self.is_falcon_dataset:
            self.internal_format_sample_indices = [self.internal_format_sample_indices[i] for i in indices_to_keep]
        else:
            self.json_paths = [self.json_paths[i] for i in indices_to_keep]

    def __len__(self):
        if self.is_falcon_dataset:
            return len(self.internal_format_sample_indices)
        else:
            return len(self.json_paths)

    @property
    def modality_lengths(self):
        return [1 for _ in range(len(self))] 

    def get_sample_at_index(self, index):
        if self.is_falcon_dataset:
            assert INTERNAL_IMPORTED
            video_bytes, json_dict = video_caption_dataset_utils.get_sample_at_idx(self, index)
        else:
            dataset_name, relpath = self.json_paths[index]
            json_path = osp.join(self.dataset_paths[dataset_name], relpath)

            video_path = json_path.replace(".json", ".mp4")
            assert osp.exists(video_path), f"Video not found: {video_path}"

            with open(video_path, 'rb') as fh:
                video_bytes = fh.read()
            
            with open(json_path, 'r') as fh:
                json_dict = json.load(fh)

        caption, video_id, _ = self.parse_json(json_dict)

        video_file_obj = io.BytesIO(video_bytes)
        return video_file_obj, caption, video_id

    def parse_json(self, content):
        # WebVid10M json field names: ['caption', 'videoid']
        # Panda70M json field names: ['caption', 'video_id', 'matching_score', 'duration', 'timestamp']
        caption = content['caption']
        if 'video_id' in content:  # Panda70M
            video_id = content['video_id']
        elif 'videoid' in content:  # WebVid10M
            video_id = content['videoid']
        else:
            raise ValueError(f"Neither 'videoid' nor 'video_id' field found in JSON dict: {content}")

        score = content.get("matching_score", 1.0)

        return caption, video_id, score

    def __getitem__(self, index):
        try:
            video_file_obj, caption, video_id = self.get_sample_at_index(index)

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
            # ======================================================================================================
            video_processor = self.data_args.video_processor
            video = video_processor.load_video(video_file_obj, self.data_args.num_frames)  # list of PIL images

            if self.data_args.image_aspect_ratio == 'pad':
                video = [expand2square(image, tuple(int(x * 255) for x in video_processor.image_mean)) for image in video]
                image_size = video[0].size
                video = video_processor.preprocess(video, return_tensors='pt')  # [T, C, H, W]

            elif self.data_args.image_aspect_ratio == "anyres":
                raise NotImplementedError("Anyres not implemented for video inputs")
            else:
                image_size = video[0].size
                video = video_processor.preprocess(video, return_tensors='pt')

            sources = preprocess_multimodal([conversation], self.data_args)
            data_dict = preprocess(sources, self.tokenizer, has_image=True)
            # ==========================================================================================================

            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])
            # image exist in the data
            data_dict['image'] = video
            data_dict['image_size'] = image_size

            return data_dict

        except Exception as e:
            # raise e
            print(f'Error processing video ID {video_id}: {e}')
            return self.__getitem__(random.randint(0, self.__len__() - 1))
