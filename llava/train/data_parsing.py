from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import transformers

from llava.constants import IGNORE_INDEX
from llava.model import *
from llava.train.video_datasets import VideoCaptionDataset
from llava.train.vicas_dataset import ViCaSDataset
from llava import distributed_utils as dist_utils


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            image_sizes = [instance['image_size'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
            batch['image_sizes'] = image_sizes

        if 'seg_frames' in instances[0]:
            seg_masks = []
            seg_frames = []
            seg_meta = []

            for sample in instances:
                seg_frames.append(sample['seg_frames'])
                seg_meta.append(sample['seg_meta'])

                if sample.get('seg_masks', None) is None:
                    seg_masks.append(None)
                else:
                    seg_masks.append(sample['seg_masks'])

        else:
            seg_masks = seg_frames = seg_meta = None

        batch.update({
            "seg_masks": seg_masks,
            "seg_frames": seg_frames,
            "seg_meta": seg_meta
        })

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    if data_args.training_data_type == "video_caption":
        train_dataset = VideoCaptionDataset(
            tokenizer=tokenizer,
            data_args=data_args
        )

    elif data_args.training_data_type == "vicas":
        train_dataset = ViCaSDataset(
            tokenizer=tokenizer,
            data_args=data_args
        )

    else:
        raise ValueError(f"Invalid training_data_type: {data_args.training_data_type}")

    assert 0.0 < data_args.subsample_factor <= 1.0
    if data_args.subsample_factor < 1.0:
        n_keep = int(round(data_args.subsample_factor * len(train_dataset)))
        indices = torch.linspace(0, len(train_dataset)-1, n_keep).round().long().tolist()
        dist_utils.print_once(f"Keeping {n_keep}/{len(train_dataset)} training samples")
        train_dataset.filter_indices(indices)

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)
