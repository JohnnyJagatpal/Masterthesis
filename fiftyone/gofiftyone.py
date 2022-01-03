
import fiftyone as fo
import fiftyone.zoo as foz



dataset = foz.load_zoo_dataset(
    "coco-2017",
    splits=["train","validation"],
    label_types=["detections","segmentations"],
    classes=["banana", "apple", "orange"],
)
session = fo.launch_app(dataset)