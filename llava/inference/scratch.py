import os
import os.path as osp
import json
import cv2

from tqdm import tqdm
from glob import glob
from llava.paths import Paths
import llava.inference.visualization as viz


def viz_pred_masks():
    gt_jsons_dir = "/mnt/bn/aliathar-yg2/datasets/videonet_training/Ours/captions_and_masks"
    pred_outputs_dir = "/mnt/bn/aliathar-yg/video_llava_output/dummy_ft/inference/Ours/pred_captions_and_masks_val"
    output_base_dir = "/mnt/bn/aliathar-yg/video_llava_output/dummy_ft/inference/Ours/pred_masks_val_viz"

    pred_files = sorted(glob(osp.join(pred_outputs_dir, "*.json")))
    for f in tqdm(pred_files, leave=False):
        with open(f, 'r') as fh:
            pred_content = json.load(fh)

        video_id = pred_content["video_id"]
        gt_jsons = glob(osp.join(gt_jsons_dir, f"{video_id:06d}*.json"))
        assert len(gt_jsons) == 1
        with open(gt_jsons[0], 'r') as fh:
            gt_content = json.load(fh)

        viz_images, filenames = viz.viz_pred_and_gt_masks(pred_content, gt_content, show_pbar=True)

        for i, image_seq in enumerate(viz_images):
            output_dir = osp.join(output_base_dir, f"{video_id:06d}", f"referral_{i}")
            os.makedirs(output_dir, exist_ok=True)

            for fname, image_t in zip(filenames, image_seq):
                cv2.imwrite(osp.join(output_dir, fname), image_t)

            with open(osp.join(output_dir, "language.txt"), 'w') as fh:
                fh.write("Prompt: " + gt_content["object_referrals"][i]["prompt"] + "\n")
                fh.write("Caption: " + gt_content["caption_raw_en"] + "\n")
                fh.write("Caption (GPT): " + gt_content["caption_raw_en_gpt"] + "\n")


if __name__ == '__main__':
    viz_pred_masks()
