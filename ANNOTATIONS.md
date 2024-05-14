# How to annotate video sequences

## Install the necessary stuff

I assume that you are on a Unix-based machine (i.e. Mac or Linux) and that you have access to a terminal. I'm used to `conda` so I will outline the steps using this tool. However, you can also work with pip (and venv) if you prefer.

First, install miniconda:
[https://docs.anaconda.com/free/miniconda/miniconda-install/](https://docs.anaconda.com/free/miniconda/miniconda-install/)

Then, create a virtual environment for the project and install some python packages.

```shell
conda create --name nbc python=3.10
conda install pytorch torchvision opencv scikit-image
```

To extract frames from video sequences, I use ffmpeg and hence I will provide the guidelines for this tool. But if you want to go with something else you are more comfortable with, feel free to do so !

You can find ffmpeg [here](https://www.ffmpeg.org/download.html)

Finally, you should `git clone` the CRAFT repository:

```shell
git clone https://github.com/clovaai/CRAFT-pytorch.git
```

## Extracting frames from video sequences

Here is the command you should use to extract frames from video sequences:

```shell
ffmpeg -i {video_sequence_file} {folder_with_frames}/{sequence_name}-frame%04d.png
```
**WARNING:** The `{folder_with_frames}` should exist before running the command (i.e. `mkdir {folder_with_frames}`

This command will process the given `video_sequence_file` and output frames in the folder `folder_with_frames`. Each frame will have the name `{sequence_name}-frame`, followed by a 4-digit number representing the frame number. Note also that the folder does not need to exist *before* the command is run (i.e. it will be automatically created).

Here's a concrete example with the video sequence `IMG_1622.MOV`:

```shell
ffmpeg -i IMG_1622.MOV 1622/1622-frame%04d.png
```

This will create a folder named `1622`, which will contain image files named `1622-frame0001.png`, `1622-frame0002.png` and so on until `1622-frame0713.png`.


## Running CRAFT on the frames

Be sure to first go the project folder: `CRAFT-pytorch`

Before being able to use the CRAFT model, there are 2 lines to delete in the file `basenet/vgg16_bn.py`. The lines to be deleted are the ones containing the deprecated `model_urls` function (line 7 and 25):

```python
import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision import models
from torchvision.models.vgg import model_urls <- DELETE THIS ONE
```

```python 
class vgg16_bn(torch.nn.Module):
    def __init__(self, pretrained=True, freeze=True):
        super(vgg16_bn, self).__init__()
        model_urls['vgg16_bn'] = model_urls['vgg16_bn'].replace('https://', 'http://') <- DELETE THIS ONE
        vgg_pretrained_features = models.vgg16_bn(pretrained=pretrained).features
```
  
Now save the file without these two lines (don't change its name !) and you're almost ready to go. The final step consists in downloading the pre-trained weights

* General: [here](https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ)
* IC15: [here](https://drive.google.com/open?id=1i2R7UIUqmkUtF0jv_3MXTqmQ_9wuAnLf)

You should now have two files, named `craft_mlt_25k.pth` and `craft_ic15_20k.pth` containing the weights.

Finally, run (one of) the CRAFT model(s) on the extracted frames:

```shell
python test.py --trained_model=craft_mlt_25k.pth --test_folder={folder_with_frames} --cuda False
```

Now get a cup of coffee and wait until the job is done ;) On my relatively old Macbook Pro, it takes ~4 seconds to process a single image !

#### The results

After running CRAFT on your folder, you will see a new directory called `result`. For each processed frame, 3 files have been created, all with the prefix `res_`

* A .jpg file: the frame with red bounding boxes around the detected text areas
* A _mask.jpg: the mask containing the character occurence probability map
* A .txt file: contains the coordinates of the bounding boxes

### Eliminate irrelevant / false detections

Since CRAFT is a text area detector, it will not only detect the panels, but also other area containing text. For this reason,
the detection should be validated. For this purpose, a script has been made to help you sort annotations (i.e. keep the relevant ones
and eliminate the others). You should launch the script like this:

```shell
python src/visualization/show_annotations.py {folder_with_frames} {result_CRAFT} {corrected_annotations} {images_with_detection}
```

where:

* `{folder_with_frames}` is the directory containing the extracted frames
* `{result_folder_from_CRAFT}` is the directory containing the CRAFT result (typically `result`, located where the previous script was launched)
* `{corrected_annotations}` is a folder where the new, corrected annotations will be written
* `{images_with_detection}` is a folder where images with relevant detections overlayed will be stored

When launching the script, the following will happen:

1. the image, with the first detection overlayed is displayed. You should look at the detection and decide if it's relevant or not
2. now close the image (hit the "q" key) and go back to the terminal.
3. answer the prompt whether you would like to keep the detection or not
4. the image, with the second detection overlayed is displayed, and the process is the same
5. this is done until all CRAFT detections have been shown
6. at the end, you will see the image with just the relevant detection
7. this image will be saved in the `{images_with_detection}` directory (and the corresponding annotations in the `{corrected_annotations}` directory)

Now, another image will be displayed (by default, the step in frame processing is 10, but you can modify this by launching the script with the `--step` option, i.e. :

```shell
python src/visualization/show_annotations.py --step 5 {folder_with_frames} {result_CRAFT} {corrected_annotations} {images_with_detection}
```

Note also that you can stop the script (i.e. CTRL+C): what you have done will still be saved (and skipped the next time you relaunch the same command)


## Add "missed" detections with CVAT

Since CRAFT is not perfect, it will also miss some panels. So, to get annotations for those missed panels, you should head over to CVAT and create an annotation job. I'll let you go back to Clement's email from April 19th and the link to the tutorial to set this up (i.e. accont creation, job creation, etc ...).

**A few important things though:**

- The images you should upload are the ones in the `{images_with_detection}` folder created before. Here you already have the detections from CRAFT outlined (in green), so you don't need to annotate the already detected panels.
- For annotations, you must use the "polygon" with 4 points: it's better than the bounding box, since it allows non-rectangular regions

Now, I let you go through all the images of the sequence, and draw nice polygons around not-detected panels ;) Note that you don't need to be super precise (as you may have seen, CRAFT is not that precise on panels, and biased towards the inside).

Once you annotated all the missed panels in all the frames, you should got to "Menu / Export job dataset". Choose the "CVAT for images 1.1" export format and click OK. This will save an XML file with quite a lot of information

### Creating annotation file from CVAT's XML file

What we want here is to have, for each image, the same kind of annotation file as we built before (when correcting CRAFT detections). To do so, you can use the following script:

```shell
python src/data/parse_cvat_annotations.py {cvat_annotations.xml} {cvat-annotations}
```

where:

* `{cvat_annotations.xml}` is the XML file you just exported
* `{cvat-annotations}` is a folder with the annotations as txt file for each image


## Building the final annotations

Now that we have both the (corrected) CRAFT annotations and the CVAT annotations, we should merge them to reach a final annotation file for each image. This is done with the following script:

```shell
python src/data/aggregate_annotations.py {craft-corrected-annotations} {cvat-annotations} {final-annotations}
```

where:

* `{craft-corrected-annotations}` is the first annotation folder created from CRAFT annotations 
* `{cvat-annotations}` is the second folder created from CVAT annotations

And {final-annotations} is the folder containing the final annotations !









