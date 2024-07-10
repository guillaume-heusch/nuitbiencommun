# Data preparation

## Put all training data in a single folder

To make things as simple as possible, all training data (i.e. images + annotations) should be located in the same folder.
To do so and to avoid unecessary space consumption, the data will be symlinked thanks to the following script:

```shell
python src/tools/symlink_annotated_data.py {dir-with-frames} {dir-with-annotations} {training-dir}
```

where:
- `dir-with-frames` is the directory containing **all** frames (as extracted with ffmpeg)
- `dir-with-annotations` is the directory containing the annotations (polygons at the moment)
- `training-dir` is the final training folder, where subdirectories `images` and `annotations` will be created

## Transforming training annotations

Depending on the used model for the detection (i.e. YOLO, Fast-RCNN, or semantic segmentation), annotations will have to 
be transformed in a suitable format. Here I describe what I did for semantic segmentation. In this case, the annotation
is a black and white image (i.e. a mask) where the panels to be detected are white, and the "background" is black.

To go from the "raw" annotations (i.e. polygons) to a mask image, you should use the following script:

```shell
python src/tools/create_mask_image_from_polygons.py {training-dir} {segmentation-training-dir}
```

where:
- `training-dir` is the training folder created above, with subdirectories `images` and `annotations` 
- `segmentation-training-dir` is the training directory for segmentation. It also contains the subdirectories `images` and `annotations`, but in this case, `annotations` consists in mask images.




