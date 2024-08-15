import cv2
import hydra
from matplotlib import pyplot as plt
from omegaconf import DictConfig

import torch
import numpy as np
from matplotlib.patches import Rectangle


from src.engine.fasterrcnn_module import FasterRCNNModule

@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="fasterrcnn_predictor"
)
def run_segmentation(cfg: DictConfig):
    """
    Perform the segmentation of panels  

    """
    # load model
    detector = FasterRCNNModule.load_from_checkpoint(cfg.model.ckpt_file)
    detector.cpu()
    detector.eval()
    detector.double()

    # load image
    image_raw = cv2.imread(cfg.image_filename)
    image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
    image = image_raw / 255.
    image = image.astype(dtype="double")
    print(image.dtype)
    image = np.moveaxis(image, 2, 0)
    h, w = image.shape[:2]
    image_tensor = torch.from_numpy(image) 
    image_batch = image_tensor.unsqueeze(0)

    predictions = detector.model(image_batch)
    boxes = predictions[0]["boxes"].detach().numpy()

    if cfg.plot:
        f, ax = plt.subplots(1, figsize=(16, 9))
        ax.imshow(image_raw)
        for i in range(boxes.shape[0]):
            b = boxes[i]
            rect = Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], edgecolor='red', facecolor='none')
            ax.add_patch(rect)
        plt.show()



if __name__ == "__main__":
    run_segmentation()
