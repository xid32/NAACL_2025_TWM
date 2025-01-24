import logging
import os
import sqlite3
from os.path import basename

import numpy as np
import tqdm

from dataset.base_dataset import ImageVideoBaseDataset
from dataset.utils import load_anno, pre_text
from dataset.video_utils import VIDEO_READER_FUNCS
from utils.distributed import is_main_process

logger = logging.getLogger(__name__)


def get_anno_by_id(cur: sqlite3.Cursor, id: int):
    """TODO: Docstring for get_anno_by_id.

    Args:
        cur (sqlite3.Cursor): The dataset cursor.
        id (int): The annotation id.

    Returns:

    """
    pass


class SQLiteImgTxtRetTrainDataset(ImageVideoBaseDataset):
    media_type = "image"

    def __init__(self, ann_file, transform, has_multi_vision_gt=False):
        super().__init__()

        self.media_type = "video" if len(ann_file) == 3 and ann_file[2] == "video" else "image"
        self.label_file, self.data_root = ann_file[:2]

        self.con = sqlite3.connect("file:" + self.label_file + "?mode=ro", uri=True)
        self.cur = self.con.cursor()

        # enable this will get stuck on NFS.
        # self.cur.execute("PRAGMA temp_store = MEMORY")
        # self.cur.execute("PRAGMA mmap_size = 30000000000")

        self.transform = transform
        # each caption has multiple image as ground_truth, e.g., ssv2
        self.has_multi_vision_gt = has_multi_vision_gt
        assert not self.has_multi_vision_gt

        self.num_examples = self.get_length()

    def get_anno(self, index):
        query = f"SELECT * FROM annos WHERE id = {index};"
        res = self.cur.execute(query)
        id, filename, caption = res.fetchone()
        anno = {"image": os.path.join(self.data_root, filename), "caption": caption}
        return anno

    def get_length(self):
        """get the number of examples in this dataset.
        Returns:

        """
        num_rows = self.cur.execute("SELECT COUNT(*) FROM annos").fetchone()[0]
        return num_rows

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):

        try:
            ann = self.get_anno(index)
            image, index = self.load_and_transform_media_data(index)
            caption = pre_text(ann["caption"])
            # key = ann["caption"] if self.has_multi_vision_gt else basename(ann["image"])
            return image, caption, index
        except Exception as e:
            index = np.random.randint(0, len(self))
            return self.__getitem__(index)


class SQLiteVidTxtRetTrainDataset(SQLiteImgTxtRetTrainDataset):
    media_type = "video"

    def __init__(
        self,
        ann_file,
        transform,
        num_frames=4,
        video_reader_type="decord",
        sample_type="rand",
        num_tries=3,
        is_paragraph_retrieval=False,
        has_multi_vision_gt=False,
    ):
        super().__init__(ann_file, transform, has_multi_vision_gt)
        self.num_frames = num_frames
        self.video_reader_type = video_reader_type
        self.video_reader = VIDEO_READER_FUNCS[video_reader_type]
        self.sample_type = sample_type
        self.num_tries = num_tries
        self.is_paragraph_retrieval = is_paragraph_retrieval

        if is_paragraph_retrieval:
            raise ValueError(f"not implemented")
