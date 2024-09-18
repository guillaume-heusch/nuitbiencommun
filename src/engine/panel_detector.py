from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from omegaconf import DictConfig

from src.engine.fasterrcnn_module import FasterRCNNModule


class PanelDetector:
    """
    Class responsible to perform the detection of panels

    User has access to:

    - .run_on_one_image
    - .show_predictions
    - .get_predictions

    Attributes
    ----------
    cfg: DictConfig
        the configuration
    detector: FasterRCNNModule
        the detector
    predictions: list
        list of predictions

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
        self.detector = FasterRCNNModule.load_from_checkpoint(
            self.cfg.model.ckpt_file
        )
        self.detector.cpu()
        self.detector.eval()
        self.detector.double()
        self.predictions = None

    def run_on_one_image(self, image: np.ndarray) -> list:
        """
        Run the panel detection on the provided image.
        Image is supposed to be RGB and in the [0-255] range.

        Parameters
        ----------
        image: np.ndarray

        """
        image_to_display = image.copy()
        image = image / 255.0
        image = np.moveaxis(image, 2, 0)  # HxWxC -> CxHxW
        image_tensor = torch.from_numpy(image)
        image_batch = image_tensor.unsqueeze(0)  # add batch dimension

        # Note the [0] at the end: there is only one image in the batch
        self.predictions = self.detector(image_batch)[0]
        if self.cfg.show_all_predictions:
            self.show_predictions(image_to_display, "All predictions")
        self.predictions = self._keep_best_predictions(self.predictions)
        self.show_predictions(image_to_display, "Best predictions")
        self.predictions = self._non_max_suppression(self.predictions)

    def get_predictions(self) -> dict:
        """
        Returns predictions

        Returns
        -------
        list:
            the list of predictions

        """
        return self.predictions

    def _keep_best_predictions(self, predictions: dict) -> dict:
        """
        Keep the best predictions: score is above the given threshold.

        TODO: there must be a simpler way ...

        Parameters
        ----------
        predictions: dict
            The predictions, as returned by the model
        score_threshold: float
            the score threshold (between 0 and 1)

        Returns
        -------
        dict:
            The predictions to keep

        """
        boxes = predictions["boxes"]
        labels = predictions["labels"]
        scores = predictions["scores"]
        all_predictions = zip(boxes, labels, scores)

        predictions_to_keep = {}
        predictions_to_keep["boxes"] = []
        predictions_to_keep["labels"] = []
        predictions_to_keep["scores"] = []
        for index, (box, label, score) in enumerate(all_predictions):
            if score > self.cfg.score_threshold:
                predictions_to_keep["boxes"].append(box)
                predictions_to_keep["labels"].append(label)
                predictions_to_keep["scores"].append(score)

        predictions_to_keep["boxes"] = torch.vstack(
            predictions_to_keep["boxes"]
        )
        predictions_to_keep["labels"] = torch.Tensor(
            predictions_to_keep["labels"]
        )
        predictions_to_keep["scores"] = torch.Tensor(
            predictions_to_keep["scores"]
        ).double()
        return predictions_to_keep

    def _non_max_suppression(self, predictions: dict) -> dict:
        """
        Performs non-max suppression of overalpping bounding boxes.

        Parameters
        ----------
        predictions: dict
            The predictions, as returned by the model

        Returns
        -------
        dict:
            The predictions to keep

        """
        boxes = predictions["boxes"]
        scores = predictions["scores"]
        labels = predictions["labels"]
        index_of_boxes_to_keep = torchvision.ops.nms(boxes, scores, 0.1)
        boxes = torch.index_select(
            boxes, 0, torch.LongTensor(index_of_boxes_to_keep)
        )
        scores = torch.index_select(
            scores, 0, torch.LongTensor(index_of_boxes_to_keep)
        )
        labels = torch.index_select(
            labels, 0, torch.LongTensor(index_of_boxes_to_keep)
        )

        predictions_to_keep = {}
        predictions_to_keep["boxes"] = boxes
        predictions_to_keep["labels"] = labels
        predictions_to_keep["scores"] = scores
        return predictions_to_keep

    def show_predictions(self, image: np.ndarray, plot_title="Predictions"):
        """
        Shows predictions on the image

        Note: the image is supposed to RGB in [0-255]

        Parameters
        ----------
        image: np.ndarray
            the image
        plot_title: str
            The title of the plot

        """
        boxes = self.predictions["boxes"]
        boxes = boxes.detach().numpy()
        if self.cfg.plot:
            f, ax = plt.subplots(1, figsize=(16, 9))
            ax.imshow(image)
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
        plt.title(plot_title)
        plt.show()

    def get_panels(self, image: np.ndarray) -> list:
        """
        Extract subimages of detected panels

        Parameters
        ----------
        image: np.ndarray
            The whole image

        Returns
        -------
        list:
            List of images (np.ndarray) of panels

        """
        panel_images = []
        boxes = self.predictions["boxes"]
        boxes = boxes.detach().numpy()
        for i in range(boxes.shape[0]):
            b = boxes[i].astype(np.int32)
            panel = image[b[1] : b[3], b[0] : b[2]]
            panel_images.append(panel)
        return panel_images

    def save_panels(self, panels: list):
        """
        Save the panels as images

        Parameters
        ----------
        panels: list
            List of images (np.ndarray) of panels

        """
        Path(self.cfg.output_dir).mkdir(exist_ok=True, parents=True)
        for panel_counter, panel in enumerate(panels):
            panel_filename = (
                Path(self.cfg.output_dir) / f"panel_{panel_counter:03d}.jpg"
            )
            panel = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(panel_filename), panel)
