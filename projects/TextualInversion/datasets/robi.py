# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from typing import List, Union

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from mmdet.datasets.api_wrappers import COCO
from mmdet.datasets.coco import CocoDataset


@DATASETS.register_module()
class ROBIDataset(CocoDataset):
    """Dataset for ROBI."""

    METAINFO = {
        'classes':
        ('gear', 'chrome_screw', 'zigzag', 'eye_bolt', 'dsub_connector', 'din_connector', 'tube_fitting'),
        # ('1', '2', '3', '4', '5', '6', '7'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100), (0, 80, 100)]
    }
    COCOAPI = COCO
    # ann_id is unique in coco dataset.
    ANN_ID_UNIQUE = True