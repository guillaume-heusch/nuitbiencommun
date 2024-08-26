import csv
from pathlib import Path


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


def read_annotation_file_for_detection(filename: Path) -> dict:
    """
    Reads an annotation file containing, in each line,
    a class label and the corresponding bounding box

    Parameters
    ----------
    filename: Path

    Returns
    -------
    dict:
        The "targets" dictionary, containing labels and bounding boxes
    
    """
    targets = {}
    targets["boxes"] = []
    targets["labels"] = []
    with open(filename, "r") as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            targets["labels"].append(int(row[0]))
            box = [int(row[i]) for i in range(1,5)]
            targets["boxes"].append(box)

    return targets

def convert_polygons_to_bounding_boxes(polygons: list) -> list:
    """
    Convert polygons to bounding boxes.

    Polygons are given as [x1, y1, x2, y2, x3, y3, x4, y4]
    Boxes are returned as [xmin, ymin, xmax, ymax]

    Parameters
    ----------
    polygons: list
        The list of polygons 

    Returns
    -------
    list: 
        The list of bounding boxes

    """
    boxes = []
    for p in polygons:

        xs = [int(p[i]) for i in range(len(p)) if i % 2 == 0]
        ys = [int(p[i]) for i in range(len(p)) if i % 2 == 1]

        left = np.min(xs)
        right = np.max(xs)
        top = np.min(ys)
        bottom = np.max(ys)

        if left > 0 and top > 0 and right < width and bottom < height:
            boxes.append([left, top, right, bottom])
            total_panels += 1
        else:
            logging.debug("box not considered: at the border")

    return boxes


