# How to annotate video sequences

### Install the necessary stuff

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

### Extracting frames from video sequences

Here is the command you should use to extract frames from video sequences:

```shell
ffmpeg -i {video_sequence_file} {folder_with_frames}/{sequence_name}-frame%04d.png
```

This command will process the given `video_sequence_file` and output frames in the folder `folder_with_frames`. Each frame will have the name `{sequence_name}-frame`, followed by a 4-digit number representing the frame number. Note also that the folder does not need to exist *before* the command is run (i.e. it will be automatically created).

Here's a concrete example with the video sequence `IMG_1622.MOV`:

```shell
ffmpeg -i IMG_1622.MOV 1622/1622-frame%04d.png
```

This will create a folder named `1622`, which will contain image files named `1622-frame0001.png`, `1622-frame0002.png` and so on until `1622-frame0713.png`.


### Running CRAFT on the frames

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
