import logging

import cv2
import hydra
from omegaconf import DictConfig

from src.engine.panel_detector import PanelDetector

logger = logging.getLogger("PREDICTOR")


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="detect_and_save_panels",
)
def run_detection(cfg: DictConfig):
    """
    Perform the detection of panels using a fine-tuned Faster RCNN.

    """
    logger.setLevel(level=logging.DEBUG)

    image = cv2.imread(cfg.image_filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    panel_detector = PanelDetector(cfg)
    panel_detector.run_on_one_image(image)
    panel_detector.show_predictions(image, "Final predictions")
    panels = panel_detector.get_panels(image)
    panel_detector.save_panels(panels)


if __name__ == "__main__":
    run_detection()
