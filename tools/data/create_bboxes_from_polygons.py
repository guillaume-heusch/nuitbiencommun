#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import logging
import os
from pathlib import Path

import click
import cv2
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from src.utils import convert_polygons_to_bounding_boxes, read_annotation_file

# TODO: improve logging
logging.basicConfig(level=logging.INFO)


@click.command()
@click.argument("input_data_dir", type=click.Path(exists=True))
@click.argument("output_data_dir", type=click.Path())
@click.option(
    "-e", "--ext", type=str, default=".jpg", help="extension of the image file"
)
@click.option("-v", "--verbose", count=True, help="print stuff")
@click.option("-P", "--plot", count=True, help="plot stuff")
def process(**kwargs):
    """

    Load images and corresponding annotations
    Create new annotation files with bounding boxes

    Input data directory must contain the subfolders:
        - image
        - annotations

    Bounding boxes are in the form [xmin, ymin, xmax, ymax]

    """
    data_dir = Path(kwargs["input_data_dir"])
    output_data_dir = Path(kwargs["output_data_dir"])
    image_file_extension = kwargs["ext"]
    verbose = kwargs["verbose"]
    plot = kwargs["plot"]

    # input data
    images_dir = data_dir / "images"
    annotations_dir = data_dir / "annotations"

    # output
    output_data_dir.mkdir(parents=True, exist_ok=True)
    output_images_dir = output_data_dir / "images"
    output_annotations_dir = output_data_dir / "annotations"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_annotations_dir.mkdir(parents=True, exist_ok=True)

    total_panels = 0
    for annotation_file in sorted(annotations_dir.iterdir()):

        # TODO: check if the test on symlink is really needed
        if annotation_file.is_symlink() and annotation_file.suffix == ".txt":

            if verbose:
                logging.info(f"Processing {annotation_file}")

            image_file = images_dir / annotation_file.name
            image_file = image_file.with_suffix(image_file_extension)
            image = cv2.imread(str(image_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, _ = image.shape

            polygons = read_annotation_file(annotation_file)
            if len(polygons) == 0:
                logging.debug("Skipping: no panels here !")
                continue

            if verbose:
                logging.debug(f"There are {len(polygons)} panels")

            boxes = convert_polygons_to_bounding_boxes(polygons, height, width)

            if plot:
                f, ax = plt.subplots(1, figsize=(16, 9))
                ax.imshow(image)
                for b in boxes:
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

            new_image_file = output_images_dir / annotation_file.name
            new_image_file = new_image_file.with_suffix(image_file_extension)
            new_annotation_file = output_annotations_dir / annotation_file.name
            new_annotation_file = new_annotation_file.with_suffix(".csv")

            # label is first, there is only one class, so label is always one
            with open(new_annotation_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                for box in boxes:
                    writer.writerow(["1"] + box)

            # symlink the image
            os.symlink(image_file.resolve(), new_image_file)

    if verbose:
        logging.info(f"There is a total of {total_panels} annotated panels")


if __name__ == "__main__":
    process()
