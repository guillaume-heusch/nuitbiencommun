import logging

import cv2
import hydra
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from omegaconf import DictConfig

from src.engine.compute_metrics import MetricsComputer
from src.engine.fasterrcnn_module import FasterRCNNModule
from src.utils import read_annotation_file_for_detection

logger = logging.getLogger("PREDICTOR")


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="fasterrcnn_predictor",
)
def run_detection(cfg: DictConfig):
    """
    Perform the detection of panels using a fine-tuned Faster RCNN.

    """
    logger.setLevel(level=logging.DEBUG)

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

    # only keep the best predictions
    # Note: there's one image in the batch, hence the 0
    # score_threshold = 0.3
    # predictions[0] = keep_best_predictions(predictions[0], score_threshold)

    # perform non-max suppression
    boxes = predictions[0]["boxes"]
    # scores = predictions[0]["scores"]
    # index_of_boxes_to_keep = torchvision.ops.nms(boxes, scores, 0.1)
    # boxes = torch.index_select(boxes, 0, torch.LongTensor(index_of_boxes_to_keep))

    # Compute metrics
    if cfg.annotation_filename is not None:
        targets = read_annotation_file_for_detection(cfg.annotation_filename)
        targets = [targets]
        metrics_computer = MetricsComputer()
        f_score = metrics_computer.run_on_batch(predictions, targets)
        print(f_score)

    if cfg.plot:
        boxes = boxes.detach().numpy()
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
