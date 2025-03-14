from .sam2 import SegmentationHeadSAM2


def build_segmentation_head(config, **kwargs):
    backbone = getattr(config, "seg_backbone", "facebook/sam2.1-hiera-tiny")

    if config.seg_head in (None, ""):
        return None
    elif config.seg_head == "sam2":
        return SegmentationHeadSAM2(
            variant=backbone,
            n_token_dims=config.hidden_size,
            n_vision_dims=config.mm_hidden_size,
            n_seg_queries=config.seg_num_queries
        )
    else:
        raise NotImplementedError(f"No head of type {config.seg_head}")
