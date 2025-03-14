import cv2
import os
import os.path as osp
import json
import numpy as np
import pycocotools.mask as mt
import copy

from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm

from llava.paths import Paths
from llava.inference.video_fetcher import VideoFrameFetcher
from vicas.viz_utils import create_color_map, annotate_image_instance


def viz_pred_and_gt_masks(pred_content, gt_content, video_frame_fetcher, show_pbar=True):
    video_id = pred_content["video_id"]
    assert gt_content["video_id"] == video_id
    cmap = create_color_map().tolist()

    gt_seg_dict = {segs_t["filename"]: segs_t for segs_t in gt_content["segmentations"]}
    assert len(gt_content["object_referrals"]) == len(pred_content["pred_object_referral_masks"])

    viz_images = []
    frame_filenames = []

    video_frames, meta = video_frame_fetcher.get_frames(gt_content) # video_frames: [T, C, H, W], (0-255), RGB
    video_frames = video_frames.flip([1]).permute(0, 2, 3, 1).numpy() # [T, H, W, C], BGR
    video_frames = {fname: frame for fname, frame in zip(meta['filenames'], video_frames)}

    for i in tqdm(range(len(gt_content["object_referrals"])), leave=False, disable=not show_pbar):
        track_ids = gt_content["object_referrals"][i]["track_ids"]
        viz_images.append([])

        for pred_dict_t in tqdm(pred_content["pred_object_referral_masks"][i], leave=False, disable=not show_pbar):  # iterate over time
            filename = pred_dict_t["filename"]
            # image_path = osp.join(frames_dir, filename)
            # assert osp.exists(image_path), f"Image not found at {image_path}"
            # image_pred = cv2.imread(image_path, cv2.IMREAD_COLOR)
            # image_gt = np.copy(image_pred)
            image_pred = np.copy(video_frames[filename])
            image_gt = np.copy(video_frames[filename])

            frame_filenames.append(filename)

            gt_dict_t = gt_seg_dict[filename]
            for track_id in track_ids:
                try:
                    index = gt_dict_t["track_ids"].index(track_id)
                except ValueError:
                    print(f"ERROR: {track_id} not found in seg list")
                    continue

                mask_rle = copy.deepcopy(gt_dict_t["mask_rles"][index])  # avoid modifying dict inplace
                mask_rle["counts"] = mask_rle["counts"].encode("utf-8")
                mask = mt.decode(mask_rle).astype(np.uint8)
                image_gt = annotate_image_instance(image_gt, mask, color=cmap[track_id], mask_border=3, mask_opacity=0.4)

            for j, mask_rle in enumerate(pred_dict_t["mask_rles"], 1):
                mask_rle = copy.deepcopy(mask_rle)
                mask_rle["counts"] = mask_rle["counts"].encode("utf-8")
                mask = mt.decode(mask_rle).astype(np.uint8)
                image_pred = annotate_image_instance(image_pred, mask, color=cmap[j], mask_border=3, mask_opacity=0.4)

            h, w = image_pred.shape[:2]
            if h > w:  # stack horizontally
                image_concat = np.concatenate((image_pred, image_gt), 1)
            else:  # stack vertically
                image_concat = np.concatenate((image_pred, image_gt), 0)

            viz_images[-1].append(image_concat)
    
    return viz_images, frame_filenames


def main(args):
    # gt_jsons_dir = "/mnt/bn/aliathar-yg2/datasets/videonet_training/Ours/captions_and_masks"
    # pred_outputs_dir = "/mnt/bn/aliathar-yg/video_llava_output/dummy_ft/inference/Ours/pred_captions_and_masks_val"
    # output_base_dir = "/mnt/bn/aliathar-yg/video_llava_output/dummy_ft/inference/Ours/pred_masks_val_viz"
    if not args.output_dir:
        args.output_dir = args.pred_dir + "_viz"

    if not args.gt_dir:
        args.gt_dir = Paths.vicas_annotations_dir(args.dataset_version)

    print(f"Output dir: {args.output_dir}")
    video_frame_fetcher = VideoFrameFetcher(args)

    pred_files = sorted(glob(osp.join(args.pred_dir, "*.json")))
    for f in tqdm(pred_files, leave=False):
        with open(f, 'r') as fh:
            pred_content = json.load(fh)

        video_id = pred_content["video_id"]
        gt_jsons = glob(osp.join(args.gt_dir, f"{video_id:06d}*.json"))
        assert len(gt_jsons) == 1
        with open(gt_jsons[0], 'r') as fh:
            gt_content = json.load(fh)

        viz_images, filenames = viz_pred_and_gt_masks(pred_content, gt_content, video_frame_fetcher, show_pbar=True)

        for i, image_seq in enumerate(viz_images):
            output_dir = osp.join(args.output_dir, f"{video_id:06d}", f"referral_{i}")
            os.makedirs(output_dir, exist_ok=True)

            for fname, image_t in zip(filenames, image_seq):
                cv2.imwrite(osp.join(output_dir, fname), image_t)

            with open(osp.join(output_dir, "language.txt"), 'w') as fh:
                fh.write("Prompt: " + gt_content["object_referrals"][i]["prompt"] + "\n")
                fh.write("Caption: " + gt_content["caption_raw_en"] + "\n")
                fh.write("Caption (GPT): " + gt_content["caption_raw_en_gpt"] + "\n")


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--pred_dir", "-i", required=True)
    parser.add_argument("--gt_dir", required=False)
    parser.add_argument("--output_dir", "-o", required=False)
    parser.add_argument("--use_falcon_dataset", action='store_true')
    parser.add_argument("--chunk_idx", type=int, default=0) # dummy for VideoFrameFetcher
    parser.add_argument("--dataset_version", default="0.3", required=False)

    main(parser.parse_args())
