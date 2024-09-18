import logging

import cv2
import hydra
import numpy as np
#import torch
#from matplotlib import pyplot as plt
#from matplotlib.patches import Rectangle
from omegaconf import DictConfig
#import torchvision
#from pathlib import Path
#
from src.engine.panel_detector import PanelDetector
#from src.engine.fasterrcnn_module import FasterRCNNModule
#from src.utils import read_annotation_file_for_detection
#from src.utils import keep_best_predictions

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
    panel_detector.save_panels(image)

   
if __name__ == "__main__":
    run_detection()
