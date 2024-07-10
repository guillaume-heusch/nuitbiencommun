from pathlib import Path

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig
from torch.utils.data import Dataset


class SegmentationDataLoader(Dataset):
    """
    Class defining a segmentation dataset

    It is a special case of a U-NICA dataset, where
    annotations are mask images.

    Attributes
    ----------
    cfg: DictConfig
        The configuration
    mask_list: list
        the list with all the mask filenames
    image_list: list
        the list with all the image filenames
    transform:
        the list of (albumentations) transforms

    """

    def __init__(self, cfg: DictConfig):
        """
        Init function.

        Parameters
        ----------
        cfg: DictConfig
            The configuration

        """
        self.cfg = cfg
        self.dataset_path = cfg.train_dir
        self.mask_list = list(
            (Path(self.dataset_path) / "annotations").iterdir()
        )
        image_folder = Path(self.dataset_path) / "images"
        self.image_list = [
            image_folder / Path(mask_path.stem).with_suffix(cfg.data.extension)
            for mask_path in self.mask_list
        ]
        self.transform = self._transform()

    def _transform(self) -> A.core.composition.Compose:
        """
        Defines the transformations applied for image augmentation
        during the training phase.

        Note that the image should be provided in RGB, otherwise
        it does not make sense to apply the normalization with
        the default values.

        Returns
        -------
        A.core.composition.Compose:
            The composition of the different transforms

        """
        model_input_height, model_input_width = self.cfg.data.image_size
        train_transform = A.Compose(
            [
                A.Resize(model_input_height, model_input_width),
                A.ShiftScaleRotate(
                    shift_limit=self.cfg.data.augment.shift_limit,
                    scale_limit=self.cfg.data.augment.scale_limit,
                    rotate_limit=self.cfg.data.augment.rotate_limit,
                    p=self.cfg.data.augment.ssr_probability,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=self.cfg.data.augment.brightness_limit,
                    contrast_limit=self.cfg.data.augment.contrast_limit,
                    p=self.cfg.data.augment.bc_probability,
                ),
                A.Normalize(),  # using default values
                ToTensorV2(),
            ]
        )
        return train_transform

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
        img = cv2.imread(str(self.image_list[idx]))  # image is BGR
        img = img[:, :, ::-1]  # convert to RGB

        # Make sure mask is binary
        mask = cv2.imread(str(self.mask_list[idx]), cv2.IMREAD_GRAYSCALE)
        mask[mask <= 128] = 0
        mask[mask > 128] = 1

        # Apply augmentation
        transformed = self.transform(image=img, mask=mask)
        img = transformed["image"]
        mask = transformed["mask"].int()[None, :]

        return img, mask

    def __len__(self):
        return len(self.mask_list)
