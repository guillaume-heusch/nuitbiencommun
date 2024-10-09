
=====================
Pre-annotating images
=====================

  .. note::

     These operations are optional, but can save quite some time.


  .. note::
    
     This was done at the very beginning of the project, when
     no models for panel detection were available. As of today,
     it would be better to use a pre-trained model for detecting
     panels, as explained in :ref:`annotation_faster_rcnn` 

In order to minimize the tedious operation of annotating images, you
can first apply some detection algorithm(s) on the extracted frames.
It will already detect some of the panels, but a careful curation will
be needed afterwards.

Running CRAFT on the frames
---------------------------

At the very beginning of the project, we had no dedicated models to 
detect panels. Hence, we used a generic model that detects text area in 
images, CRAFT. You can find more information on this algorithm 
`here <https://github.com/clovaai/CRAFT-pytorch>`_. 
Here's the procedure to apply this detector to the extracted frames:

  1. Clone and install the CRAFT repository:

     .. code-block:: shell

        > git clone https://github.com/clovaai/CRAFT-pytorch.git
        > cd CRAFT-pytorch
  
  2. Edit the ``basenet/vgg16_bn.py`` file, and save it:

     .. code-block:: python

        import torch
        import torch.nn as nn
        import torch.nn.init as init
        from torchvision import models
        from torchvision.models.vgg import model_urls # <- DELETE THIS ONE

  
     .. code-block:: python

        class vgg16_bn(torch.nn.Module):
          def __init__(self, pretrained=True, freeze=True):
              super(vgg16_bn, self).__init__()
              # DELETE / COMMENT OUT THE FOLLOWING LINE
              #model_urls['vgg16_bn'] = model_urls['vgg16_bn'].replace('https://', 'http://')

  3. Download the pre-trained model's weights
     `here <https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ>`_. 
     This will download a file named `craft_mlt_25k.pth`.

  4. Run the CRAFT model on the extracted frames:

     .. code-block:: shell

        python test.py --trained_model=craft_mlt_25k.pth --test_folder={folder_with_frames} --cuda False

  5. Now get a cup of coffee and wait until the job is done ;) 

  
After running CRAFT on your folder, you will see a new directory called ``result``. 
For each processed frame, 3 files have been created, all with the prefix ``res_``:

- A `.jpg` file: the frame with red bounding boxes around the detected text areas
- A `_mask.jpg` file: the mask containing the character occurence probability map
- A `.txt` file: contains the coordinates of the bounding boxes
    

Eliminate irrelevant / false detection
--------------------------------------

Since CRAFT is a text area detector, it will not only detect the panels, 
but also other areas containing text. The detections should hence be further validated. 
For this purpose, a script has been made to help you sort annotations 
(i.e. keep the relevant ones and eliminate the others). You should launch the script like this:

     .. code-block:: shell

        python tools/annotations/edit_craft_detections.py {folder_with_frames} {result_CRAFT} {corrected_annotations} {images_with_detection}

where:

* ``{folder_with_frames}`` is the directory containing the extracted frames.
* ``{result_folder_from_CRAFT}`` is the directory containing the CRAFT result.
* ``{corrected_annotations}`` is a folder where the new, corrected annotations will be written.
* ``{images_with_detection}`` is a folder where images with relevant detections overlayed will be stored

When launching the script, the following will happen:

1. the image, with the first detection overlayed is displayed. You should look at the detection and decide if it's relevant or not
2. now close the image (hit the "q" key) and go back to the terminal.
3. answer the prompt whether you would like to keep the detection or not
4. the image, with the second detection overlayed is displayed, and the process is the same
5. this is done until all CRAFT detections have been shown
6. at the end, you will see the image with just the relevant detection
7. this image will be saved in the ``{images_with_detection}`` directory (and the corresponding annotations in the ``{corrected_annotations}`` directory)

Now, the next image will be displayed, and you can keep on checking until the whole sequence has been processed.

  .. note::

     By default, the step between considered images is 10 (i.e. one image every ten is processed). 
     This has been done since consecutive frames are highly correlated: it is hence not really useful
     to have all the frames for training the model. 
     You can modify this by launching the script with the ``--step`` option

Note also that you can stop the script (i.e. CTRL+C): what you have done will still be saved 
(and skipped the next time you relaunch the same command)


  .. todo::

    This script should be refactored (functions should be put somewhere in ``src``)
