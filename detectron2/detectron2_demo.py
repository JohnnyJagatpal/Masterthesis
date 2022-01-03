import torch
import numpy as np
import os
import json
import cv2
import random


import detectron2
from detectron2 import config
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from utils.datasethelper import dataset_loader,dataset_preview


from detectron2.engine import DefaultTrainer


# Checking for installed version of torch and cuda
torch_version = ".".join(torch.__version__.split(".")[:2])
cuda_version = torch.__version__.split("+")[-1]
print("torch: ", torch_version, "\n cuda: ", cuda_version)

if not (torch_version == "1.10" and cuda_version == "cu113"):
    raise Exception("Pytorch or cuda is not matching the required version!\n \
        Pytorch 1.10 and Cuda cu113 is required")


setup_logger()

dataset_dicts, dataset_metadata = dataset_loader("../Datasets/coco-fruit/","fruit")
dataset_dict_train = dataset_dicts["train"]
dataset_dict_test = dataset_dicts["validation"]
dataset_metadata_train = dataset_metadata["train"]



for d in random.sample(dataset_dict_train,3):
    dataset_preview(d,dataset_metadata_train)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("fruit_train")
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1000
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()


cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
cfg.DATASETS.TEST = ("fruit_test", )
predictor = DefaultPredictor(cfg)

for d in random.sample(dataset_dict_test,3):
    dataset_preview(d,dataset_metadata=dataset_metadata_train,predictor=predictor)