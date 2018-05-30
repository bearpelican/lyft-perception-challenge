#!/bin/bash
# May need to uncomment and update to find current packages
apt-get update

# Required for demo script! #
pip install --upgrade pip
pip install scikit-video
pip install opencv-python

# Add your desired packages for each workspace initialization
#          Add here!          #
conda install pytorch torchvision cuda91 -c pytorch -y
# wget https://s3-us-west-2.amazonaws.com/ashaw-fastai-imagenet/600urn-36-resnet-softmax-nocrop-tmp.h5
wget https://s3-us-west-2.amazonaws.com/ashaw-fastai-imagenet/600urn-38-wide-crop-eval-nocrop-tmp-2.h5