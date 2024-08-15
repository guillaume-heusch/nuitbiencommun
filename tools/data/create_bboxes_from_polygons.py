#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import logging
import os
from pathlib import Path

import click
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

logging.basicConfig(level=logging.INFO)


def read_annotation_file(filename: Path) -> list:
    """
    Reads an annotation file containing polygons
    and returns the list of annotations for polygons

    Parameters
    ----------
    filename: Path

    Returns
    -------
    list:
        The list of polygons

    """
    polygons = []
    with open(filename, "r") as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            polygons.append(row)
    return polygons


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

    Load an image file and corresponding annotations
    Create a mask image with the panels

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

        if annotation_file.is_symlink() and annotation_file.suffix == ".txt":

            if verbose:
                logging.info(f"Processing {annotation_file}")

            image_file = images_dir / annotation_file.name
            image_file = image_file.with_suffix(image_file_extension)
            image = cv2.imread(str(image_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            img_bbox = image.copy()
            polygons = read_annotation_file(annotation_file)

            if len(polygons) == 0:
                logging.info("Skipping: no panels here")
                continue

            boxes = []
    
            if verbose:
                logging.info(f"There are {len(polygons)} panels")
                total_panels += len(polygons)
            
            height, width, _ = image.shape

            for p in polygons:

                xs = [int(p[i]) for i in range(len(p)) if i % 2 == 0]
                ys = [int(p[i]) for i in range(len(p)) if i % 2 == 1]

                left = np.min(xs)
                right = np.max(xs)
                top = np.min(ys)
                bottom = np.max(ys)
                
                #if plot:
                #    f, ax = plt.subplots(1, figsize=(16, 9))
                #    ax.imshow(image)
                #    rect = Rectangle((left, top), right-left, bottom-top, edgecolor='red', facecolor='none')
                #    ax.add_patch(rect)
                #    plt.show()


                if left > 0 and top > 0 and right < width and bottom < height:
                    boxes.append([left, top, right, bottom])
                else:
                    print("box not appended")

            if plot:
                f, ax = plt.subplots(1, figsize=(16, 9))
                ax.imshow(image)
                for b in boxes:
                    rect = Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], edgecolor='red', facecolor='none')
                    ax.add_patch(rect)
                plt.show()

            new_image_file = output_images_dir / annotation_file.name
            new_image_file = new_image_file.with_suffix(image_file_extension)
            new_annotation_file = output_annotations_dir / annotation_file.name
            new_annotation_file = new_annotation_file.with_suffix(".csv")

            # write annotation file
            # label is first, there is only one class, so label is always one
            with open(new_annotation_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                for box in boxes:
                    writer.writerow(["1"] + box)
            
            # symlink the image
            os.symlink(image_file.resolve(), new_image_file)
    
    if verbose:
        logging.info(f"There is a total of {total_panels} annotated panels")


if __name__ == "__main__":
    process()
