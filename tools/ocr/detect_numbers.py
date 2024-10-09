import logging
from pathlib import Path

import cv2
import hydra
from matplotlib import pyplot as plt
from ocr_tamil.ocr import OCR
from omegaconf import DictConfig

logger = logging.getLogger("PREDICTOR")


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="ocr_detection_on_panels",
)
def run_detection(cfg: DictConfig):
    """
    Perform digit recognition on a set of panel images

    """
    logger.setLevel(level=logging.DEBUG)

    # OCR engine
    ocr = OCR(details=1, lang=["english"])
    images_list = [i for i in Path(cfg.panels_dir).iterdir()]
    images_list.sort()

    for image_file in images_list:
        text_list = ocr.predict(str(image_file))
        result, score = text_list[0]

        # load and preprocess image
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if cfg.plot:
            plt.imshow(image)
            plt.title(result)
            plt.show()


if __name__ == "__main__":
    run_detection()
