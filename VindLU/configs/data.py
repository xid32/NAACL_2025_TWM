import os as __os  # add "__" if not want to be exported
from copy import deepcopy as __deepcopy

data_dir = __os.environ.get("VL_DATA_DIR")
if data_dir is None:
    raise ValueError("please set environment `VL_DATA_DIR` before continue")

data_root = __os.path.join(data_dir, "videos_images")
anno_root_pt = __os.path.join(data_dir, "anno_pretrain")
anno_root_downstream = __os.path.join(data_dir, "anno_downstream")

# ============== pretraining datasets=================
available_corpus = dict(
    # pretraining datasets
    cc3m=[f"{anno_root_pt}/cc3m_train.sqlite.db", f"{data_root}/cc3m_224"],
    cc12m=[f"{anno_root_pt}/cc12m.sqlite.db", f"{data_root}/cc12m_224"],
    sbu=[f"{anno_root_pt}/sbu.sqlite.db", f"{data_root}/sbu_224"],
    vg=[f"{anno_root_pt}/vg.sqlite.db", f"{data_root}/vg"],
    coco=[f"{anno_root_pt}/coco.sqlite.db", f"{data_root}/coco"],
    webvid=[f"{anno_root_pt}/webvid_train.sqlite.db", f"{data_root}/webvid_2fps_224", "video"],
    webvid_10m=[
        f"{anno_root_pt}/webvid_10m_train.sqlite.db",
        f"{data_root}/webvid_10m_2fps_224",
        "video",
    ],
    # downstream datasets.
)

# composed datasets.
available_corpus["coco_vg"] = [available_corpus["coco"], available_corpus["vg"]]
available_corpus["webvid_cc3m"] = [available_corpus["webvid"], available_corpus["cc3m"]]
available_corpus["webvid_14m"] = [
    available_corpus["webvid"],
    available_corpus["cc3m"],
    available_corpus["coco"],
    available_corpus["vg"],
    available_corpus["sbu"],
    available_corpus["cc12m"],
]
available_corpus["webvid12m_14m"] = [
    available_corpus["webvid"],
    available_corpus["webvid_10m"],
    available_corpus["cc3m"],
    available_corpus["coco"],
    available_corpus["vg"],
    available_corpus["sbu"],
    available_corpus["cc12m"],
]
available_corpus["webvid10m_14m"] = [
    available_corpus["webvid_10m"],
    available_corpus["cc3m"],
    available_corpus["coco"],
    available_corpus["vg"],
    available_corpus["sbu"],
    available_corpus["cc12m"],
]

# ============== for validation =================
available_corpus["msrvtt_1k_test"] = [
    f"{anno_root_downstream}/msrvtt_test1k.json",
    f"{data_root}/msrvtt_2fps_224",
    "video",
]

