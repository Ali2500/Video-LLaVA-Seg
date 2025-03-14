import os
import os.path as osp


def _get_env_var(varname):
    val = os.environ.get(varname, None)
    if val is None:
        raise EnvironmentError(f"Required env variable '{varname}' not set")
    return val


def _assert_exists(path):
    if not osp.exists(path):
        raise ValueError(f"Path does not exist: {path}")
    return path



class Paths:
    def __init__(self):
        raise ValueError("Static class should not be initialized")

    # @staticmethod
    # def webvid_train_hdfs_tensorbundle_dir():
    #     if 'HDFS_EU' in os.environ:
    #         return osp.join(_get_env_var('HDFS_EU'), "datasets", "WebVid10M", "tensor_bundles")
    #     else:
    #         return osp.join(_get_env_var('HDFS_SG'), "datasets", "WebVid10M_tensor_bundles")

    # @staticmethod
    # def panda70m_train_hdfs_tensorbundle_dir():
    #     return osp.join(get_hdfs_root(), "datasets", "Panda70M", "filtered_tensor_bundles")

    @staticmethod
    def saved_models_dir():
        return _assert_exists(_get_env_var("VIDEONET_MODELS_DIR"))

    # @staticmethod
    # def our_dataset_videos_dir():
    #     return _assert_exists(osp.join(_get_env_var("VIDEONET_TRAINING_DATA_2"), "Ours", "videos"))

    # @staticmethod
    # def our_dataset_video_tensorbundles():
    #     return [
    #         osp.join(get_hdfs_root(), "datasets", "Ours", "videos", "oops_tensor_bundle", "dataset"),
    #         osp.join(get_hdfs_root(), "datasets", "Ours", "videos", "remaining_tensor_bundle", "dataset")
    #     ]

    # @staticmethod
    # def our_dataset_video_frames_dir():
    #     return _assert_exists(osp.join(_get_env_var("VIDEONET_TRAINING_DATA_2"), "Ours", "video_frames"))

    # @staticmethod
    # def our_dataset_video_frames_tensorbundle_dir():
    #     return osp.join(get_hdfs_root(), "datasets", "Ours", "video_frames", "tensor_bundles")

    # @staticmethod
    # def our_dataset_captions_path():
    #     return _assert_exists(osp.join(_get_env_var("BYTENAS_ROOT"), "datasets", "video-llava_eval", "Ours", "captions.jsonl"))

    @staticmethod
    def datasets_base_dir():
        return _assert_exists(osp.realpath(osp.join(osp.dirname(__file__), osp.pardir, "datasets")))

    @staticmethod
    def vicas_base_dir():
        return _assert_exists(osp.join(Paths.datasets_base_dir(), "ViCaS"))

    @staticmethod
    def vicas_videos_dir():
        return _assert_exists(osp.join(Paths.vicas_base_dir(), "videos"))

    @staticmethod
    def vicas_video_frames_dir():
        return _assert_exists(osp.join(Paths.vicas_base_dir(), "video_frames"))

    @staticmethod
    def vicas_annotations_dir(version: str):
        return _assert_exists(osp.join(Paths.vicas_base_dir(), "annotations", version))

    @staticmethod
    def vicas_split_json(version: str, split: str):
        return _assert_exists(osp.join(Paths.vicas_base_dir(), "splits", version, f"{split}.json"))

    @staticmethod
    def mevis_base_dir():
        return _assert_exists(osp.join(Paths.datasets_base_dir(), "MeViS"))

    @staticmethod
    def revos_base_dir():
        return _assert_exists(osp.join(Paths.datasets_base_dir(), "ReVOS"))

    @staticmethod
    def webvid_train_dir():
        return _assert_exists(osp.join(Paths.datasets_base_dir(), "WebVid10M", "train"))

    @staticmethod
    def panda70m_train_dir():
        return _assert_exists(osp.join(Paths.datasets_base_dir(), "Panda70M", "train"))
