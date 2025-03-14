import decord
import torch
import torch.nn as nn
import numpy as np

from typing import List, Dict, Optional, Union
from PIL import Image
from transformers import CLIPImageProcessor
from transformers.models.clip.image_processing_clip import PILImageResampling, ChannelDimension

decord.bridge.set_bridge('torch')


class CLIPVideoProcessor(CLIPImageProcessor):
    def preprocess(self, images, *args, **kwargs):
        assert isinstance(images, (list, tuple, torch.Tensor))
        assert kwargs.get("return_tensors", "pt") == "pt"
        return torch.stack([super(CLIPVideoProcessor, self).preprocess(img, *args, **kwargs)["pixel_values"][0] for img in images])

    # def resize(
    #     self, 
    #     image: np.ndarray, 
    #     size: Dict[str, int], 
    #     resample: PILImageResampling = ..., 
    #     data_format: Optional[Union[str, ChannelDimension]] = None, 
    #     input_data_format: Optional[Union[str, ChannelDimension]] = None, 
    #     **kwargs
    # ) -> np.ndarray:
    #     return super(CLIPVideoProcessor, self).resize(image, size, resample, data_format, input_data_format, **kwargs)

    @classmethod
    def load_video(cls, video_path_or_file_obj: str, num_frames: int, return_normalized_timestamps: bool = False) -> List[Image.Image]:
        decord_vr = decord.VideoReader(video_path_or_file_obj, ctx=decord.cpu(0))
        duration = len(decord_vr)
        frame_id_list = np.linspace(0, duration-1, num_frames, dtype=int)
        video_data = decord_vr.get_batch(frame_id_list)  # [T, H, W, C]
        video_data = video_data.numpy()
        frames = [Image.fromarray(frame) for frame in video_data]
        
        if return_normalized_timestamps:
            timestamps = frame_id_list.astype(np.float32) / float(duration)
            return frames, timestamps
        else:
            return frames
