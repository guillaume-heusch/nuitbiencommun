
.. _annotation_faster_rcnn:

====================================
Pre-annotating images (Faster R-CNN)
====================================

  .. note::

     These operations are optional, but can save quite some time.


Now that models have already been trained on some data, you can use
such models to help annotating images. For this purpose, you should follow
the following procedure.

Running a pre-trained Faster R-CNN on the frames
------------------------------------------------

To run a model on a set of images in a directory, you should use the
following script:

 .. code-block:: shell

    > python tools/detection/detect_and_save_results_folder.py

This script uses a configuration file where you can specify several variables.
Here's an example:

  .. code-block:: yaml

     input_dir: "{dir-with-images}"
     image_file_extensions:
       - ".jpg"
     output_dir: "{dir-where-to-write-annotations}"
     model:
       ckpt_file: "{model-filename}.ckpt"
     score_threshold: 0.3 # must be [0-1]
     show_detections: False
     show_all_predictions: False
     plot: False
     frame_step: 10 # to process images every frame_step frames

Note that you can override the configuration on the command line, by specifying
each field. For instance, if you want to change the ``input_dir``:

  .. code-block:: shell
    
    > python tools/detection/detect_and_save_results_folder.py input_dir=another_dir_with_images
     

Eliminate irrelevant / false detection
--------------------------------------

Now that you a (possibly) rough model has been applied to your images,
you have to check the results to be sure to have valid annotations.
For this purpose, you can use the following script:

  .. code-block:: shell

    > python tools/annotations/edit_fasterrcnn_detections.py {folder_with_frames} {result_FasterRCNN} {corrected_annotations} {images_with_detections}

where:

* ``{folder_with_frames}`` is the directory containing the extracted frames.
* ``{result_FasterRCNN}`` is the directory containing the Faster R-CNN result.
* ``{corrected_annotations}`` is a folder where the new, corrected annotations will be written.
* ``{images_with_detection}`` is a folder where images with relevant detections overlayed will be stored

When launching the script, the following will happen:

1. the image, with the first detection overlayed is displayed. You should look at the detection and decide if it's relevant or not
2. now close the image (hit the "q" key) and go back to the terminal.
3. answer the prompt whether you would like to keep the detection or not
4. the image, with the second detection overlayed is displayed, and the process is the same
5. this is done until all Faster R-CNN detections have been shown
6. at the end, you will see the image with just the relevant detection
7. this image will be saved in the ``{images_with_detection}`` directory (and the corresponding annotations in the ``{corrected_annotations}`` directory)

Now, the next image will be displayed, and you can keep on checking until the whole sequence has been processed.

Note also that you can stop the script (i.e. CTRL+C): what you have done will still be saved 
(and skipped the next time you relaunch the same command)
