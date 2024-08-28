import csv
from pathlib import Path

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig
from torch.utils.data import Dataset


class DetectionDataLoader(Dataset):
    """
    Class defining a detection dataset

    Attributes
    ----------
    cfg: DictConfig
        The configuration
        the list of (albumentations) transforms

    """

    def __init__(self, cfg: DictConfig, transforms=None):
        """
        Init function.

        Parameters
        ----------
        cfg: DictConfig
            The configuration

        """
        self.cfg = cfg
        self.dataset_path = cfg.train_dir

        self.transforms = self._get_train_transform()

        self.annotations_list = list(
            (Path(self.dataset_path) / "annotations").iterdir()
        )
        image_folder = Path(self.dataset_path) / "images"
        self.images_list = [
            image_folder
            / Path(annotation_path.stem).with_suffix(cfg.data.extension)
            for annotation_path in self.annotations_list
        ]

        self.images_list.sort()
        self.annotations_list.sort()

    def _get_train_transform(self):
        """ """
        return A.Compose(
            [A.Normalize(), ToTensorV2(p=1.0)],
            bbox_params={
                "format": "pascal_voc",
                "label_fields": ["class_labels"],
            },
        )

    def __getitem__(self, idx):
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
        # image = F.to_tensor(img)

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

            # boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            image_id = torch.tensor([idx])

            if self.transforms is not None:
                transformed = self.transforms(
                    image=img, bboxes=boxes, class_labels=labels
                )
                image = transformed["image"]
                boxes = transformed["bboxes"]

            # WARNING: a check on the bounding box should be made here
            # be sure it's fully in the image for instance

            # if np.min(boxes) < 0:
            #    print(f"WARNING, something wrong with bboxes {boxes}")
            #    print(f"{annotation_path}")

            # print(boxes)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = image_id
            target["image_path"] = str(img_path)

        return image, target

    def __len__(self):
        return len(self.annotations_list)
