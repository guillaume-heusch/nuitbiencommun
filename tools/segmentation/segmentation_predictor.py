import cv2
import hydra
from matplotlib import pyplot as plt
from omegaconf import DictConfig

from src.engine.segmentation_module import SegmentationModule
from src.image_processing.normalize_image import (
    denormalize_mask_after_segmentation,
    normalize_image_for_segmentation,
)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="segmentation_predictor",
)
def run_segmentation(cfg: DictConfig):
    """
    Perform the segmentation of panels

    """
    # load model
    segmenter = SegmentationModule.load_from_checkpoint(cfg.model.ckpt_file)
    segmenter.cpu()
    segmenter.eval()

    # load image
    image = cv2.imread(cfg.image_filename)
    h, w = image.shape[:2]
    image = image[:, :, ::-1]  # convert to RGB

    # transformations to apply to the image
    img = normalize_image_for_segmentation(image, cfg.image_size)

    logits = segmenter.model(img)
    pr_masks = logits.sigmoid()
    mask = (pr_masks >= 0.5).float()
    # resize mask back to original image size
    mask = denormalize_mask_after_segmentation(mask, [h, w])

    if cfg.plot:
        f, ax = plt.subplots(1, 2)
        ax[0].imshow(image)
        ax[1].imshow(mask, cmap="gray")
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        plt.show()


if __name__ == "__main__":
    run_segmentation()
