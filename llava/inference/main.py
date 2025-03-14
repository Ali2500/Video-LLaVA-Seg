import math
import os
import os.path as osp
import argparse
import json
import warnings
import io
import random
import string
import torch
import transformers
import warnings

# warnings.filterwarnings("ignore", category=UserWarning, module=osp.join(osp.dirname(torch.__file__), "torch.nn*"))
# warnings.filterwarnings("ignore", category=UserWarning, module=osp.join(osp.dirname(transformers.__file__), "transformers.generation*"))

import torch
from glob import glob
from tqdm import tqdm

from llava import conversation
from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import (
    DEFAULT_IMAGE_TOKEN, 
    IMAGE_TOKEN_INDEX, 
    DEFAULT_VID_START_TOKEN, 
    DEFAULT_VID_END_TOKEN,
    DEFAULT_SF_VID_SEPARATOR_TOKEN
)
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.model.language_model.llava_llama import LlavaConfig
from llava.utils import disable_torch_init
from llava.paths import Paths

from llava.inference.video_fetcher import VideoAndFrameFetcher
from llava.inference.utils import preprocess_seg_inputs, mask_tensor_to_rle, get_null_mask_rle
from llava.utils import torch_to


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def get_video_list(gt_dir: str, version: str, split: str, output_dir: str, overwrite: bool, chunk_idx: int, num_chunks: int):
    with open(Paths.vicas_split_json(version, split), 'r') as fh:
        split_video_ids = json.load(fh)

    print(f"Processing {len(split_video_ids)} videos")
    split_video_ids = set(get_chunk(split_video_ids, num_chunks, chunk_idx))

    completed_video_ids = set()
    for f in glob(osp.join(output_dir, "*.json")):
        completed_video_ids.add(int(osp.split(f)[-1].replace(".json", "")))

    json_files = sorted(glob(osp.join(gt_dir, "*.json")))
    ret = []
    for f in json_files:
        video_id = int(osp.split(f)[-1].replace(".json", ""))
        if video_id not in split_video_ids:
            continue

        split_video_ids.remove(video_id)
        if video_id in completed_video_ids and not args.overwrite:
            continue
        ret.append(f)

    if split_video_ids:
        raise FileNotFoundError(f"No JSON file found for the following {len(split_video_ids)} videos: {sorted(list(split_video_ids))}")
    
    return ret


def get_model_output(
        model, 
        video_processor, 
        tokenizer, 
        seg_frames,
        seg_meta,
        video_filehandle, 
        qs, 
        conv_template, 
        args,
    ):
    # num_frames = model.config.num_frames
    if model.config.num_slow_frames == model.config.num_frames:
        num_slow_frames = model.config.num_frames
        num_fast_frames = 0
        sf_separator = ""
    else:
        num_slow_frames = model.config.num_slow_frames
        num_fast_frames = model.config.num_frames
        sf_separator = DEFAULT_SF_VID_SEPARATOR_TOKEN if model.config.mm_use_sf_vid_separator_token else ""

    # vid_replace_token = DEFAULT_IMAGE_TOKEN * num_frames_per_video
    vid_tokens = (DEFAULT_IMAGE_TOKEN * num_slow_frames) + sf_separator + (DEFAULT_IMAGE_TOKEN * num_fast_frames)

    if model.config.mm_use_im_start_end:
        qs = DEFAULT_VID_START_TOKEN + vid_tokens + DEFAULT_VID_END_TOKEN + qs
    else:
        qs = vid_tokens + qs

    conv = conv_template.copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    video_frames = video_processor.load_video(video_filehandle, model.config.num_frames) 
    video_tensor = video_processor.preprocess(video_frames, return_tensors='pt')
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
    image_size = video_frames[0].size

    input_ids = input_ids.to(device='cuda', non_blocking=True)
    video_tensor = video_tensor.to(dtype=args.dtype, device='cuda', non_blocking=True)
    seg_meta = torch_to(seg_meta, dtype=args.dtype, device='cuda')

    with torch.inference_mode():
        output_dict = model.generate(
            input_ids,
            images=[video_tensor],
            image_sizes=[image_size],
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
            seg_frames=seg_frames,
            seg_meta=seg_meta,
        )

    output_ids = output_dict['sequences']
    output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return output_text, output_dict["pred_mask_logits"].cpu() if "pred_mask_logits" in output_dict else None


def run_inference(args):
    if not osp.isabs(args.model_path): # and not args.model_path.startswith("fun-research"):
        args.model_path = osp.join(Paths.saved_models_dir(), args.model_path)
        assert osp.exists(args.model_path), f"Model path not found at {args.model_path}"

    if not args.dtype:
        config = LlavaConfig.from_pretrained(args.model_path)
        if config.mm_vision_tower.startswith("nvidia/"):
            args.dtype = 'bfloat16'  # RADIO encoder doesn't work with float16
        else:
            args.dtype = 'float16'

        if args.chunk_idx == 0:
            print(f"Setting dtype to {args.dtype}")

    video_and_frame_fetcher = VideoAndFrameFetcher(args)

    if not args.gt_dir:
        args.gt_dir = Paths.vicas_annotations_dir(args.dataset_version)

    if not args.output_dir:
        args.output_dir = f"inference/ViCaS/pred_{args.dataset_version}_{args.dataset_split}"

    if not osp.isabs(args.output_dir):
        args.output_dir = osp.join(args.model_path, args.output_dir)
    
    print(f"Output directory: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.dtype == 'float16':
        args.dtype = torch.float16
    elif args.dtype == 'bfloat16':
        args.dtype = torch.bfloat16
    elif args.dtype == 'float32':
        args.dtype = torch.float32
    else:
        raise ValueError("Invalid dtype: ", args.dtype)

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = "llava-v1.6" + get_model_name_from_path(model_path) # hack to get into the correct if-else segments in `model/builder.py`
    tokenizer, model, video_processor, context_len = load_pretrained_model(
        model_path=model_path, 
        model_base=None, 
        model_name=model_name,
        dtype=args.dtype
    )
    model = model.to(args.device)
    config = model.config

    gt_json_list = get_video_list(
        gt_dir=args.gt_dir,
        version=args.dataset_version,
        split=args.dataset_split,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        chunk_idx=args.chunk_idx,
        num_chunks=args.num_chunks
    )

    if not gt_json_list:
        print(f"No videos to process")
        exit(0)

    if config.use_text_prompt:
        caption_prompt = "\nDescribe what is happening in the video in detail. Talk about the events, the main objects/actors and also briefly describe the background scene."
    else:
        caption_prompt = ""

    # conv_template_name = getattr(config, "conversation_template", "llava_llama_3")
    conv_template_name = config.conversation_template
    conv_template = conv_templates[conv_template_name]

    # pred_captions = []

    pbar = tqdm(gt_json_list, leave=False, disable=args.chunk_idx > 0)
    for json_path in pbar:
        with open(json_path, 'r') as fh:
            content = json.load(fh)

        video_id = content['video_id']

        pbar.set_description(str(video_id))
        output_path = osp.join(args.output_dir, f"{video_id:06d}.json")
        if osp.exists(output_path) and not args.overwrite:
            continue

        # load video
        video_name = content['filename']
        video_bytes = video_and_frame_fetcher.get_video(video_name)

        # load video frames for segmentation
        seg_frames, seg_meta = video_and_frame_fetcher.get_frames(
            json_content=content, 
            # max_seg_frames=config.max_seg_frames,
            gt_only=True
        )

        seg_frames, seg_meta = preprocess_seg_inputs(
            seg_frames=seg_frames,
            seg_meta=seg_meta,
            video_processor=video_processor,
            tgt_size=config.seg_image_size,
            normalize=False,
            pad_mode=config.seg_pad_mode
        )
        seg_frames = seg_frames.to(dtype=args.dtype, device=args.device)

        video_pred_dict = {
            "video_id": video_id,
            "pred_lgvis_masks": [[] for _ in range(len(content["object_referrals"]))],
            "pred_caption": None
        }

        sub_indices = []
        if not args.skip_captions:
            sub_indices.append(-1)
        if not args.skip_seg:
            sub_indices.extend(list(range(len(content["object_referrals"]))))

        for idx in tqdm(sub_indices, leave=False, disable=True):
            if idx == -1:  # captioning task
                prompt = caption_prompt
                kwargs = {'seg_frames': None, 'seg_meta': None}
            else:  # object referral
                prompt = f"\n{content['object_referrals'][idx]['prompt']} Please output the segmentation mask."
                kwargs = {'seg_frames': [seg_frames], 'seg_meta': [seg_meta]}

            pred_text, pred_mask_logits = get_model_output(
                model=model, 
                video_processor=video_processor, 
                tokenizer=tokenizer, 
                video_filehandle=io.BytesIO(video_bytes), 
                qs=prompt, 
                conv_template=conv_template, 
                args=args, 
                **kwargs
            )

            if not pred_text and idx == -1:
                print(f"WARN: Predicted caption for video {video_id} is a null string")

            if args.print_captions and idx == -1:
                print(f"Video {video_id}\nPred Caption: {pred_text}\nGT Caption: {content['caption_parsed_en_gpt']}\n-------------------------------")

            if idx != -1:
                # print(pred_mask_logits.min().item(), pred_mask_logits.max().item())
                pred_masks = pred_mask_logits > 0.0  # [N, T, H, W]
                num_tracks, num_frames = pred_masks.shape[:2]
                assert tuple(pred_masks.shape[-2:]) == tuple(seg_meta['orig_image_size']), f"Spatial size mismatch: {pred_masks.shape}, {seg_meta['orig_image_size']}"
                assert num_frames == seg_frames.shape[0]
                assert num_frames == len(seg_meta["filenames"])

                pred_mask_rles = mask_tensor_to_rle(pred_masks)  # Outer list: number of objects, inner list: number of frames

                for t in range(num_frames):
                    segs_t = {
                        "filename": seg_meta["filenames"][t],
                        "mask_rles": [x[t] for x in pred_mask_rles],
                    }
                    video_pred_dict["pred_lgvis_masks"][idx].append(segs_t)

            else:
                video_pred_dict["pred_caption"] = pred_text

        with open(output_path, 'w') as fh:
            json.dump(video_pred_dict, fh)


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--model_path', '-i', help='', required=True)
    parser.add_argument('--gt_dir', help='Path to GT dir', required=False)
    parser.add_argument('--output_dir', "-o", help='Path of directory for storing results.', required=False)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--dtype", required=False) # , default='float16', choices=('float16', 'bfloat16', 'float32'))
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--overwrite", action='store_true')
    parser.add_argument("--print_captions", action='store_true')
    # --------
    parser.add_argument("--use_internal_loader", action='store_true')
    parser.add_argument("--dataset_version", default="v1.0")
    parser.add_argument("--dataset_split", default="val")
    parser.add_argument("--skip_seg", action='store_true')
    parser.add_argument("--skip_captions", action='store_true')
    # --------
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
