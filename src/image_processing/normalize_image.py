import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2


def denormalize_mask_after_segmentation(mask, resize_dim: list) -> np.ndarray:
    """
    Convert a binary mask of type torch.Tensor into a binary uint8 image as numpy array

    Parameters
    ----------
    mask: torch.Tensor
        The binary mask as Tensor with dimension (1, 1, H, W)
    resize_dim: list
        Size of the image (height, width) to create
    Returns
    -------
    numpy.ndarray:
        The mask with original image size as uint8 monochanel image

    """
    # resize mask back to original image size
    mask = torch.squeeze(mask)
    mask = mask.detach().numpy() * 255
    mask = mask.astype(np.uint8)
    return cv2.resize(mask, (resize_dim[1], resize_dim[0]))


def normalize_image_for_segmentation(image: np.ndarray, resize_dim: list):
    """
    normalize an RGB image for segmentation using SegmentationModule

    Parameters
    ----------
    image: numpy.ndarray
        The (color) image for which the mask is computed.
    resize_dim: list
        Size of the image (height,width) to create
    Returns
    -------
    torch.Tensor:
        Noralized image for DeepLabV3Plus module

    """
    transform = A.Compose(
        [
            A.Resize(resize_dim[0], resize_dim[1]),
            A.Normalize(),
            ToTensorV2(),
        ]
    )
    transformed = transform(image=image)
    img = transformed["image"]
    img = img[None, :]  # add minibatch dimension
    return img
