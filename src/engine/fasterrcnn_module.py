import logging

import numpy as np
import pytorch_lightning as pl

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead

class FasterRCNNModule(pl.LightningModule):
    """
    """
    def __init__(self, pretrained_weights_path=None):
        super().__init__()
        self.model = self._get_model(num_classes=2)
        self.validation_f_score = []


    def _get_model(self, num_classes: int, pretrained=True):
        """
        Returns the model defined the config params
        """

        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=pretrained, pretrained_backbone=False
        )

        # number of different feature map sizes output by the backbone model
        n_features_map_sizes = 5

        # custom anchor generator, tailored for our specific problem: small characters
        anchor_generator = AnchorGenerator(
            sizes=tuple([(8, 16, 32) for _ in range(n_features_map_sizes)]),
            aspect_ratios=tuple(
                [(1.0, 1.5, 3.0) for _ in range(n_features_map_sizes)]
            ),
        )
        model.rpn.anchor_generator = anchor_generator

        # 256 because that's the number of features that FPN returns
        model.rpn.head = RPNHead(
            256, anchor_generator.num_anchors_per_location()[0]
        )

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        return model

    def forward(self, image):
        """Runs inference."""
        self.model.eval()
        output = self.model(image)
        return output

    def training_step(self, batch, batch_idx):
        image, target = batch
        loss_dict = self.model(image, target)
        losses = sum(loss for loss in loss_dict.values())

        batch_size = len(batch[0])
        self.log_dict(loss_dict, batch_size=batch_size)
        self.log("train_loss", losses, batch_size=batch_size)

        return losses

    def validation_step(self, batch, batch_idx):
        image, target = batch
        output = self.model(image)
        f_scores = self.evaluate()


    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        return [optimizer], [lr_scheduler]

    def evaluate(self, dataloader, device=torch.device('cpu')):
        """
        Evaluate the model on a dataset

        Parameters
        ----------
        dataloader: :py:class:`torch.utils.data.DataLoader`
          The dataloader for your validation data
        device: :py:class:`torch.torch.device`
          The device where the training is performed
        verbosity: int
          Verbosity level

        """
        self.network.eval()
        true_positive = false_positive = false_negative = 0
        for index, (image, target, _) in enumerate(dataloader):
          if self.verbosity > 1:
            print("Evaluating image {}/{}".format(index+1, len(dataloader)))
          predictions = self.network(image)
          stats = get_tp_fp_fn(image[0], predictions, target[0], verbosity=self.verbosity, plot=0)
          for k, v in stats.items():
            true_positive += v[0]
            false_positive += v[1]
            false_negative += v[2]

        F1_score = compute_F1_score(true_positive, false_positive, false_negative)
        return F1_score
        
