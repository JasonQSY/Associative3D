
'''
'''
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
from tqdm import tqdm
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from dataloader import get_dicts
from detectron2.data.datasets import register_coco_instances
import cv2
import random

# for d in ["train", "validation", "test"]:
#     register_coco_instances("suncg/"+d,{}, 'v3_'+d+'.json', '/w/syqian/suncg/renderings_ldr')
#     MetadataCatalog.get("suncg/"+d).set(evaluator_type='coco')

nyu_json_path = '/y/jinlinyi/nyu_test/detectron_nyu.json'
nyu_img_path = '/y/jinlinyi/nyu_test/pngs'
# register_coco_instances("nyu",{}, nyu_json_path, nyu_img_path)
# suncg_metadata = MetadataCatalog.get("nyu").set(evaluator_type='coco')

cfg = get_cfg()
cfg.merge_from_file("../../detectron2/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
# cfg.MODEL.WEIGHTS = "/w/jinlinyi/faster_rcnn_models/model_0044999.pth"  # initialize from model zoo
cfg.MODEL.WEIGHTS = 'model_final_68b088.pkl'
save_dir = '/y/jinlinyi/nyu_test/debug'

os.makedirs(save_dir, exist_ok=True)
cfg.NUM_GPUS=1
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
cfg.DATASETS.TEST = ("nyu", )
predictor = DefaultPredictor(cfg)
from detectron2.utils.visualizer import ColorMode
from detectron2.data.datasets import load_coco_json
dataset_dicts = load_coco_json(nyu_json_path, nyu_img_path, dataset_name="nyu")
suncg_metadata = MetadataCatalog.get("nyu")#.set(evaluator_type='coco')
# dataset_dicts = get_dicts("test")
#  = MetadataCatalog.get("nyu")
# evaluators = DatasetEvaluators(COCOEvaluator("suncg/validation", cfg, True, save_dir),)
# res = cls.test(cfg, model, evaluators)
# import pdb;pdb.set_trace()
for idx, d in enumerate(tqdm(random.sample(dataset_dicts, 300))):    
    im = cv2.imread(d["file_name"])
    # im = im[:, :, ::-1]
    outputs = predictor(im)
    # import pdb;pdb.set_trace()
    v = Visualizer(im[:, :, ::-1],
                   metadata=suncg_metadata, 
                   scale=0.8, 
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite(os.path.join(save_dir, f'{idx}.png'), v.get_image()[:, :, ::-1])