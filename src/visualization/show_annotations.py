#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import logging
import os
import sys

import click
import cv2
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon

import numpy as np

@click.command()
@click.argument("images-dir")
@click.argument("annotations-dir")
def process(**kwargs):
    """
    Show annotations

    """
    images_dir = kwargs["images_dir"]
    annotations_dir = kwargs["annotations_dir"]

    images = os.listdir(images_dir)
    images.sort()

    for i, img_file in enumerate(images):
        logging.info(
            "Processing {} ({}/{})".format(img_file, i + 1, len(images))
        )

        img_file = os.path.join(images_dir, img_file)
        stem, ext = os.path.splitext(img_file)
        try:
            image = cv2.imread(img_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except cv2.error:
            logger.error("Unable to load image {}".format(img_file))
            sys.exit()

        csv_file = "res_" + os.path.basename(img_file)
        csv_file = csv_file.replace(ext, ".txt")
        csv_file = os.path.join(annotations_dir, csv_file)
        print(csv_file)

        try:
            detections = [] 
            with open(csv_file) as csvfile:
                reader = csv.reader(csvfile)
                for row in enumerate(reader):
                    detections.append([int(x) for x in row[1]])
        except FileNotFoundError:
            logger.warning("No annotations for {}".format(img_file))
            continue

        ## show annotations
        fig, ax = plt.subplots(1, figsize=(16, 9))
        ax.imshow(image)
        ax.set_title(img_file)
        ax.set_xticks([])
        ax.set_yticks([])
        for line in detections:
            x = [line[i] for i in range(len(line)) if i % 2 == 0]
            y = [line[i] for i in range(len(line)) if i % 2 == 1]
            xy = np.column_stack((x, y)) 
            poly = Polygon(
                xy,
                ec="springgreen",
                lw=2,
                facecolor="none",
            )
            ax.add_patch(poly)
        plt.show()


if __name__ == "__main__":
    process()
