import csv
import random
from pathlib import Path
from typing import Tuple

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig
from torch.utils.data import Dataset


def get_train_and_valid_lists(cfg: DictConfig) -> Tuple[list, list]:
    """
    Parse the directory with data and split into train
    and validation subsets

    Parameters
    ----------
    cfg: DictConfig
        The configuration

    Returns
    -------

    """
    dataset_path = cfg.train_dir
    annotations_list = list((Path(dataset_path) / "annotations").iterdir())
    annotations_list.sort()
    random.seed(cfg.seed)
    random.shuffle(annotations_list)

    dataset_size = len(annotations_list)
    train_size = int(cfg.data.train_ratio * dataset_size)
    annotations_list_train = annotations_list[:train_size]
    annotations_list_valid = annotations_list[train_size:]

    # TODO: use logger instead of print
    print(f"train dataset size = {len(annotations_list_train)}")
    print(f"validation dataset size = {len(annotations_list_valid)}")

    return annotations_list_train, annotations_list_valid


def get_train_transform():
    """
    Defines the augmentations for the training data.

    TODO: normalization parameters should be computed on the training set

    Returns
    -------
    albumentations.core.composition.Compose
        The transformations.

    """
    return A.Compose(
        [
            # A.RandomSizedBBox(min_area=0.1, max_area=1.0, p=0.5),
            # A.RandomBrightnessContrast(p=0.2),
            # A.HueSaturationValue(p=0.2),
            # A.RGBShift(p=0.2),
            # A.RandomGamma(p=0.2),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.1),
            # A.Rotate(limit=10, p=0.2),
            # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.2),
            # A.Resize(height=800, width=800),
            # WARNING: don't use ImageNet's normalization parameters !!!
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc", label_fields=["class_labels"]
        ),
    )


def get_valid_transform():
    """
    Defines the augmentations for the validation data

    TODO: normalization parameters should be computed on the training set

    Returns
    -------
    albumentations.core.composition.Compose
        The transformations.

    """
    return A.Compose(
        [
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc", label_fields=["class_labels"]
        ),
    )


class DetectionDataLoader(Dataset):
    """
    Class defining a detection dataset

    Attributes
    ----------
    cfg: DictConfig
        The configuration
        the list of (albumentations) transforms
    annotations_list: list
        The list of annotation files
    dataset_path: str
        Path to the training (and validation) data

    """

    def __init__(self, cfg: DictConfig, annotations_list, transforms=None):
        """
        Init function.

        Parameters
        ----------
        cfg: DictConfig
            The configuration
        annotations_list: list
            The list of annotation files
        transforms: albumentations.core.composition.Compose
            The transformations to be applied to the image

        """
        self.cfg = cfg
        self.annotations_list = annotations_list
        self.transforms = transforms
        self.dataset_path = cfg.train_dir

        image_folder = Path(self.dataset_path) / "images"
        self.images_list = [
            image_folder
            / Path(annotation_path.stem).with_suffix(cfg.data.extension)
            for annotation_path in self.annotations_list
        ]
        self.images_list.sort()
        self.annotations_list.sort()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        """
        Returns image and mask for the selected index as torch tensors.

        Parameters
        ----------
        idx: int
            The selected index

        Returns
        -------
        torch.Tensor:
            the image
        torch.Tensor:
            the corresponding mask

        """
        img_path = self.images_list[idx]
        img = cv2.imread(str(img_path))  # image is BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        annotation_path = self.annotations_list[idx]
        assert annotation_path.stem == img_path.stem

        with open(annotation_path) as csvfile:
            reader = csv.reader(csvfile)
            target = {}
            boxes = []
            labels = []
            for index, row in enumerate(reader):
                boxes.append(
                    [int(row[1]), int(row[2]), int(row[3]), int(row[4])]
                )
                labels.append(int(row[0]))

            labels = torch.as_tensor(labels, dtype=torch.int64)
            image_id = torch.tensor([idx])

            if self.transforms is not None:
                transformed = self.transforms(
                    image=img, bboxes=boxes, class_labels=labels
                )
                image = transformed["image"]
                boxes = transformed["bboxes"]

            # WARNING: a check on the bounding box should be made here
            # EDIT: made when transforming polygons to bounding boxes
            boxes = torch.as_tensor(boxes, dtype=torch.float32)

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = image_id
            target["image_path"] = str(img_path)

        return image, target

    def __len__(self) -> int:
        """
        Returns the length of the dataset

        """
        return len(self.annotations_list)
