"""
Function and classes for easy work with datasets
"""

from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
import cv2
import matplotlib.pyplot as plt



def dataset_loader(path,name="dataset",split=["train","validation"],coco=True):
    """
    Provides the selected parts of the dataset.
    Returns two dictionaries.
    A dictonary containing the dataset dictonaries for each split.
    A dictonary containing the dataset metadata for each split.
    """

    dataset_dicts = {}
    dataset_metadata ={}

    # registering the dataset in coco format
    if coco is True:

        for d in split:
            register_coco_instances(f"{name}_{d}", {}, f"{path}{d}/labels.json", f"{path}{d}/data")
            dataset_dicts[d] = DatasetCatalog.get(f"{name}_{d}")
            dataset_metadata[d] = MetadataCatalog.get(f"{name}_{d}")
    else:
        raise Exception("Not implemented")
        #   TODO Implement other datasettypes


    return dataset_dicts,dataset_metadata
    

def dataset_preview(sample,dataset_metadata=None,predictor=None):
    """
    Shows the selected image with the corresponding segmentation.
    """
    img = cv2.imread(sample["file_name"])
    v = Visualizer(img[:, :, ::-1], metadata=dataset_metadata, scale=0.5,instance_mode=ColorMode.IMAGE_BW)
    if predictor is None:  
        v = v.draw_dataset_dict(sample)
    else:
        outputs = predictor(img)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))   
    plt.figure(figsize=(14, 10))
    plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1],cv2.COLOR_BGR2RGB))