import logging

import cv2
import hydra
from omegaconf import DictConfig

from src.engine.panel_detector import PanelDetector

logger = logging.getLogger("PREDICTOR")


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="detect_and_save_results_folder",
)
def run_detection(cfg: DictConfig):
    """
    Perform the detection of panels using a fine-tuned Faster RCNN.

    """
    logger.setLevel(level=logging.DEBUG)

    panel_detector = PanelDetector(cfg)
    panel_detector.run_on_dir()


if __name__ == "__main__":
    run_detection()
