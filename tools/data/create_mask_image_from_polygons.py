#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
from pathlib import Path

import click
import cv2
import numpy as np
from matplotlib import pyplot as plt

from src.utils import read_annotation_file

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

    Load image files and corresponding annotations
    Create mask images with the panels

    Input data directory must contain the subfolders:
        - image
        - annotations

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
    output_images_dir.mkdir(parents=True, exist_ok=False)
    output_annotations_dir.mkdir(parents=True, exist_ok=True)

    for annotation_file in sorted(annotations_dir.iterdir()):

        # TODO: check if the test on symlink is really needed
        if annotation_file.is_symlink() and annotation_file.suffix == ".txt":

            if verbose:
                logging.info(f"Processing {annotation_file.name}")

            image_file = images_dir / annotation_file.name
            image_file = image_file.with_suffix(image_file_extension)
            image = cv2.imread(str(image_file))

            img_poly = image.copy()
            polygons = read_annotation_file(annotation_file)
            mask = np.zeros_like(image, dtype=np.uint8)

            for p in polygons:

                xs = [p[i] for i in range(len(p)) if i % 2 == 0]
                ys = [p[i] for i in range(len(p)) if i % 2 == 1]

                points_list = []
                for x, y in zip(xs, ys):
                    points_list.append([int(x), int(y)])
                points = np.array(points_list)
                cv2.polylines(img_poly, [points], True, (0, 0, 255), 1)
                cv2.fillPoly(mask, [points], (255))

            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask[np.where(mask > 0)] = 255

            if plot:
                f, ax = plt.subplots(1, 3, figsize=(16, 9))
                res = cv2.bitwise_and(image, image, mask=mask)
                ax[0].imshow(image)
                ax[1].imshow(res)
                ax[2].imshow(mask, cmap="gray")
                plt.show()

            new_image_file = output_images_dir / annotation_file.name
            new_image_file = new_image_file.with_suffix(image_file_extension)
            new_annotation_file = output_annotations_dir / annotation_file.name
            new_annotation_file = new_annotation_file.with_suffix(
                image_file_extension
            )

            cv2.imwrite(str(new_annotation_file), mask)
            os.symlink(image_file.resolve(), new_image_file)


if __name__ == "__main__":
    process()
