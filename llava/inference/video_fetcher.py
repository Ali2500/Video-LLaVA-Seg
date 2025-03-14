from PIL import Image
from llava.paths import Paths
from typing import Dict, List, Any, Optional, Tuple

import einops
import io
import numpy as np
import os.path as osp
import random
import string
import torch

try:
    from llava.internal.video_fetcher import VideoAndFrameFetcher
    INTERNAL_IMPORTED = True
except ImportError as _:
    INTERNAL_IMPORTED = False


class _VideoAndFrameFetcher:
    def __init__(self, args):
        assert not args.use_internal_loader
        self.video_dir = Paths.vicas_videos_dir()
        self.video_frames_dir = Paths.vicas_video_frames_dir()

    def get_video(self, filename):
        video_path = osp.join(self.video_dir, filename)
        assert osp.exists(video_path), f"Video not found at {video_path}"
        with open(video_path, 'rb') as fh:
            video_bytes = fh.read()

        return video_bytes

    def get_frames(
            self, 
            json_content: Dict[str, Any],
            max_seg_frames: int = -1,
            gt_only: bool = True
        ):
        frames = []
        filenames = []
        frame_indices = []
        meta = {}

        for t, segs_t in enumerate(json_content["segmentations"]):
            if gt_only and not segs_t["is_gt"]:
                continue

            image_path = osp.join(self.video_frames_dir, f"{json_content['video_id']:06d}", segs_t['filename'])
            assert osp.exists(image_path), f"Video frame not found: {image_path}"
            with open(image_path, 'rb') as fh:
                image_bytes = fh.read()

            filenames.append(segs_t['filename'])

            image_bytes = io.BytesIO(image_bytes)
            image = np.array(Image.open(image_bytes))
            h, w = image.shape[:2]
            meta['orig_image_size'] = (h, w)
            frames.append(image)
            frame_indices.append(t)

        frames = torch.from_numpy(np.stack(frames))  # [T, H, W, 3], RGB, [0-255]
        frames = einops.rearrange(frames, "T H W C -> T C H W")
        frame_indices = torch.tensor(frame_indices, dtype=torch.float32) / float(len(json_content["segmentations"]) - 1)

        if frames.shape[0] > max_seg_frames and max_seg_frames > 0:
            keep_indices = torch.linspace(0, frames.shape[0]-1, max_seg_frames, dtype=torch.long).round().to(device=frames.device)
            frames = frames[keep_indices]
            filenames = [filenames[i] for i in keep_indices.tolist()]

        meta.update({
            'video_id': json_content['video_id'],
            'filenames': filenames,
            'timestamps': frame_indices
        })

        return frames, meta

    def get_random_string(self, length: int):
        return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))


if not INTERNAL_IMPORTED:
    VideoAndFrameFetcher = _VideoAndFrameFetcher
