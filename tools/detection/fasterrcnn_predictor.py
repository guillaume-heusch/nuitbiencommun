import cv2
import hydra
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from omegaconf import DictConfig

from src.engine.fasterrcnn_module import FasterRCNNModule


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="fasterrcnn_predictor",
)
def run_detection(cfg: DictConfig):
    """
    Perform the detection of panels using a fine-tuned Faster RCNN.

    """
    # load model
    detector = FasterRCNNModule.load_from_checkpoint(cfg.model.ckpt_file)
    detector.cpu()
    detector.eval()
    detector.double()

    # load and preprocess image
    image_raw = cv2.imread(cfg.image_filename)
    image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
    image = image_raw / 255.0
    image = np.moveaxis(image, 2, 0)
    image_tensor = torch.from_numpy(image)
    image_batch = image_tensor.unsqueeze(0)

    predictions = detector.model(image_batch)
    # there's one image in the batch, hence the 0
    boxes = predictions[0]["boxes"].detach().numpy()

    if cfg.plot:
        f, ax = plt.subplots(1, figsize=(16, 9))
        ax.imshow(image_raw)
        for i in range(boxes.shape[0]):
            b = boxes[i]
            rect = Rectangle(
                (b[0], b[1]),
                b[2] - b[0],
                b[3] - b[1],
                edgecolor="red",
                facecolor="none",
                linewidth=2,
            )
            ax.add_patch(rect)
        plt.show()


if __name__ == "__main__":
    run_detection()
