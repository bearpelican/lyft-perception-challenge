#!/bin/bash
# May need to uncomment and update to find current packages
# apt-get update

# Required for demo script! #
pip install scikit-video

# Add your desired packages for each workspace initialization
#          Add here!          #
conda install pytorch torchvision cuda91 -c pytorch
wget https://s3-us-west-2.amazonaws.com/ashaw-fastai-imagenet/600urn-36-resnet-softmax-nocrop-tmp.h5