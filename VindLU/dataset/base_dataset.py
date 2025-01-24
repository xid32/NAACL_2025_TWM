import logging
import random

from torch.utils.data import Dataset

from dataset.utils import load_image_from_path

logger = logging.getLogger(__name__)


class ImageVideoBaseDataset(Dataset):
    """Base class that implements the image and video loading methods"""

    media_type = "video"

    def __init__(self):
        assert self.media_type in ["image", "video"]
        self.data_root = None
        self.anno_list = (
            None  # list(dict), each dict contains {"image": str, # image or video path}
        )
        self.transform = None
        self.video_reader = None
        self.num_tries = None

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def get_anno(self, index):
        """obtain the annotation for one media (video or image)

        Args:
            index (int): The media index.

        Returns: dict.
            - "image" or "video": the filename.
            - "caption": The caption for this file.

        """
        anno = self.anno_list[index]
        if self.data_root is not None:
            anno[self.media_type] = os.path.join(self.data_root, anno[self.media_type])
        return anno

    def load_and_transform_media_data(self, index):
        if self.media_type == "image":
            return self.load_and_transform_media_data_image(index)
        else:
            return self.load_and_transform_media_data_video(index)

    def load_and_transform_media_data_image(self, index):
        ann = self.get_anno(index)
        data_path = ann["image"]
        image = load_image_from_path(data_path)
        image = self.transform(image)
        return image, index

    def load_and_transform_media_data_video(self, index):
        for i in range(self.num_tries):
            ann = self.get_anno(index)
            data_path = ann["image"]
            try:
                max_num_frames = self.max_num_frames if hasattr(self, "max_num_frames") else -1
                frames, frame_indices, video_duration = self.video_reader(
                    data_path, self.num_frames, self.sample_type, max_num_frames=max_num_frames
                )
            except Exception as e:
                index = random.randint(0, len(self) - 1)
                logger.warning(
                    f"Caught exception {e} when loading video {data_path}, "
                    f"randomly sample a new video as replacement"
                )
                continue

            frames = self.transform(frames)
            return frames, index
        else:
            raise RuntimeError(
                f"Failed to fetch video after {self.num_tries} tries. "
                f"This might indicate that you have many corrupted videos."
            )
