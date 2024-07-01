#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
from pathlib import Path

import click

logging.basicConfig(level=logging.INFO)


@click.command()
@click.argument("frames_dir", type=click.Path(exists=True))
@click.argument("annotations_dir", type=click.Path(exists=True))
@click.argument("destination_dir", type=click.Path())
@click.option(
    "-e", "--ext", type=str, default=".png", help="extension of the image file"
)
@click.option("-v", "--verbose", count=True, help="print stuff")
def process(**kwargs):
    """

    Go through the frames and anotations dir, and
    symlink pairs in destination_dir/images and destination_dir/annotations

    """
    frames_dir = Path(kwargs["frames_dir"])
    annotations_dir = Path(kwargs["annotations_dir"])
    destination_dir = Path(kwargs["destination_dir"])
    image_file_extension = kwargs["ext"]
    verbose = kwargs["verbose"]

    destination_dir_images = destination_dir / "images"
    destination_dir_annotations = destination_dir / "annotations"
    destination_dir_images.mkdir(exist_ok=True, parents=True)
    destination_dir_annotations.mkdir(exist_ok=True, parents=True)

    for src_annotation_file in sorted(annotations_dir.iterdir()):

        if (
            src_annotation_file.is_file()
            and src_annotation_file.suffix == ".txt"
        ):
            src_image_file = frames_dir / src_annotation_file.name
            src_image_file = src_image_file.with_suffix(image_file_extension)

            dst_annotation_file = (
                destination_dir_annotations / src_annotation_file.name
            )
            dst_annotation_file = dst_annotation_file.with_suffix(".txt")
            dst_image_file = destination_dir_images / src_image_file.name
            dst_image_file = dst_image_file.with_suffix(image_file_extension)

            if verbose:
                logging.info(f"linking {src_image_file} to {dst_image_file}")
                logging.info(
                    f"linking {src_annotation_file} to {dst_annotation_file}"
                )
                logging.info("-" * 50)
            os.symlink(src_annotation_file, dst_annotation_file)
            os.symlink(src_image_file, dst_image_file)


if __name__ == "__main__":
    process()
