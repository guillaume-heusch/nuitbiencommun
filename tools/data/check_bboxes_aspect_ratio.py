#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture

from src.utils import read_annotation_file_for_detection

# TODO: improve logging
logging.basicConfig(level=logging.INFO)


@click.command()
@click.argument("annotations_dir", type=click.Path(exists=True))
@click.option(
    "-n",
    "--ngaussians",
    type=int,
    default=3,
    help="Number of components in GMM",
)
@click.option(
    "-e",
    "--ext",
    type=str,
    default=".csv",
    help="extension of the annotation file",
)
@click.option("-v", "--verbose", count=True, help="print stuff")
def process(**kwargs):
    """
    Load annotations and check bounding boxes aspect ratio

    Bounding boxes are in the form [xmin, ymin, xmax, ymax]

    """
    annotations_dir = Path(kwargs["annotations_dir"])
    annotation_file_extension = kwargs["ext"]
    n_gaussians = kwargs["ngaussians"]
    verbose = kwargs["verbose"]

    aspect_ratios = []
    for annotation_file in sorted(annotations_dir.iterdir()):

        if annotation_file.suffix == annotation_file_extension:

            if verbose:
                logging.info(f"Processing {annotation_file}")

            targets = read_annotation_file_for_detection(annotation_file)
            boxes = targets["boxes"]
            for xmin, ymin, xmax, ymax in boxes:
                width = xmax - xmin
                height = ymax - ymin
                aspect_ratios.append(height / float(width))

    aspect_ratios = np.array(aspect_ratios).reshape(-1, 1)

    # fit a GMM on aspect ratios (this will give the means)
    gmm = GaussianMixture(n_components=n_gaussians)
    gmm.fit(aspect_ratios)

    # Print the (interesting) model parameters
    print("Means:\n", gmm.means_)
    print("\nWeights:\n", gmm.weights_)

    # plot histgoram and GMM
    x = np.linspace(aspect_ratios.min(), aspect_ratios.max(), 1000).reshape(
        -1, 1
    )
    logprob = gmm.score_samples(x)
    responsibilities = gmm.predict_proba(x)
    pdf = np.exp(logprob)
    pdf_individual = responsibilities * pdf[:, np.newaxis]
    plt.figure(figsize=(10, 6))
    plt.hist(aspect_ratios, bins=30, edgecolor="green", density=True)
    plt.plot(x, pdf, "-k", label="GMM")
    plt.plot(x, pdf_individual, "--", label="Individual components")
    plt.title("Gaussian Mixture Model on aspect ratio")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    process()
