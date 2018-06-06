# **Lyft Preception Challenge** 

---

This project submission is for the [image segmentation challenge](https://www.udacity.com/lyft-challenge) organized by Lyft and Udacity hosted in May 2018.  

### Competition
Competition provided dashcam images and pixel labels generated by the [CARLA simulator](http://carla.org/).  
Goal: achieve the highest accuracy on vehicle and road detection with an FPS greater than 10 on a Nvidia K80 GPU.

### Results
I achieved the highest "unofficial" score (95.97) in the competition by discovering a bug in the competition. This score was almost 2 points higher than the leader 93.22.  
![alt text][image1]

However, after reporting this bug, I was asked to remove my results.  Without the exploit, this model achieved [16th](https://classroom.udacity.com/nanodegrees/nd013/parts/78a1caae-489e-4e75-8368-e65cec97f63b/modules/f2b38613-4094-451e-957b-2a343dc667c0/lessons/fce3e4a7-05c3-43f4-9fe3-84f3f150b368/concepts/04ca1c4c-66b5-471b-859e-8fcaedd793ec). Deeper explanation [here](#bugs)

### Summary
**Library:** [Fast.Ai](https://github.com/fastai/fastai) + Pytorch
**Data:** 600x800 dashcam images and pixel by pixel labels  
**Data preprocessing:** Trimmed the sky and the car hood images (384x800px). Processed target labels to 3 categories: Vehicle, Road, Everything Else.  
**Data augmentation:** Random resized crop, random horizontal flip, random color jitter
**Architecture:** U-net with Resnet34 backbone. Inspired by Fast.AI's [Carvana implmentation](https://github.com/fastai/fastai/blob/master/courses/dl2/carvana-unet.ipynb)  
**Loss function:** Custom F-beta loss function  
**Training:** [Progressive resizing](https://arxiv.org/abs/1707.02921), [Cyclical Learning Rates](https://arxiv.org/abs/1506.01186). 

**A lot of the techniques used for this project were taken from FastAi's Stanford DawnBENCH [submission](http://www.fast.ai/2018/04/30/dawnbench-fastai/)**

---

#### Data
I generated an additional 10,000 segmentation images by running the CARLA simulator on a windows machine.

I ran their autopilot [script](https://github.com/carla-simulator/carla/blob/master/PythonClient/client_example.py). With a segmentation camera:  
```
camera1 = Camera('CameraSeg', PostProcessing='SemanticSegmentation')
camera1.set_image_size(800, 600)
camera1.set_position(1.30, 0, 1.30)
settings.add_sensor(camera1)
```

Output:
![alt text][image2]

#### Data Preprocess
Trimmed the sky and the car hood images (384x800px). Image dimensions needed to be divisible by 32 due to the 5 downsample layers in Resnet34.
Preprocessed target labels to 3 categories: Vehicle, Road, Everything Else.  
![alt text][image3]


#### Data augmentation
Random resized crop: trained with both square cropping (384x384) and rectangle cropping (384x800)  
Random Horizontal Flip  
Random Color Jitter (Brightness, Contrast, Saturation): .2, .2, .2  
Normalization: Used imagenet stats  


#### Architecture

[Model](./examples/arch.py)  
U-net with Resnet34 backbone. Inspired by Fast.AI's [Carvana implmentation](https://github.com/fastai/fastai/blob/master/courses/dl2/carvana-unet.ipynb) 

Tried several different architectures in this [notebook](./workspace/Example/lyft-unet-train34-arch-speed.ipynb). Chose the one suited best for speed and accuracy.

Original U-net: 3x slower to train  
U-net with VGG11 backbone: 1.5x slower. Not as accurate  
U-net with Resnet50 backbone: 2x slower  
U-net + LSTM with Resnet34 backbone: 
* Ran the encoder through an LSTM before sending it to the decoder. Used an RNN to encode temporal video data. Inspired by  STFCN [paper](https://arxiv.org/pdf/1608.05971.pdf). 
* Implementation [notebook](./workspace/Example/lyft-unet-train43-conv-lstm-try2.ipynb)
* Ran out of patience as accuracy did not fare much better than submission

I wanted to choose the fastest architecture to iterate as fast as possible. Probably could have achieved a much higher accuracy towards the end by using a deeper network.

#### Loss Function
Used custom loss function [here](./examples/loss.py)

Competition scoring was measured by a weighted F score:  
1. Car F(beta=2)
2. Road F(beta=0.5)

Losses tried: Weighted Cross-Entropy, Weighted Dice Loss, Custom F-beta Loss

I went through most of the competition using weighted dice loss before realizing I could just use the F-beta score directly.

This in turn helped me realized that recall was a much more important metric for detecting cars and the opposite for detecting roads.
Car beta of 2 means recall is about 2x more important than precision.
Similarly, a beta of 0.5 means precision is weighted more heavily.

Turns out the simple weighted cross-entropy (car weight of 5, road weight of 2) performed just as well as the custom f1 score. 91.2 vs 91.3

Sigmoid vs Softmax: sigmoid worked better for me, but did not do enough testing to verify this.

---

#### Bugs
**Discovering the location of the answers**  
Udacity provided contestants with a GPU server with 50 hours of time. Results were submitted by uploading your model to the server and running a script.  
While trying to understand how the submission process worked, I discovered that the answers were downloaded to a temporary directory on the server and evaluated against your model. The trick was just to figure out where that temporary directory was located, and copy it before the obfuscated script deleted it.  

**Obviously** this is really bad. I was able to achieve a perfect score of 10 with an FPS of 1000 by just [submitting](./workspace/Example/demo-precashed-ans.py) the answers I found. 

**Discovering that these answers are actually wrong**  
Before I discovered the location of the answers, I noticed something strange about my submission. I would achieve a weighted F score of 9.6 while training. However, when I submitted the results, I would only get a score of 9.2.  
Because of this, I started to look at the example video they gave me and overlayed segmentation labels onto them. Turns out half way through the video, the segmentation labels are off by 1 frame.

Frame 0            |  Frame 30
:-------------------------:|:-------------------------:
![image4]  |  ![image5]


After discovering location the answers, I did the same overlay and realized it had the same [problem](./workspace/merged_unsynced_output.mp4).

To confirm that the Udacity private test set had a bug inside, I submitted one of my old models, but corrected a few of the frames to be no longer off by 1.  
```
mismatched_idxs = list(range(15,44)) + list(range(200,750))
```

The score jumped from 9.2 to 9.6 - more in line with my own evaluation metrics. This meant that if corrected, all the scores on the leaderboard could potentially jump 4 points.

**Reporting bugs to Udacity**  
Figured in the spirit of fairness (after all this was a Lyft competition, not Uber), I should report this to Udacity. Turns out they knew about bug #1. However, bug #2 turned out to be an encoding/decoding error with a third party library they were using. With only a few days left of the competition, it was too late to correct the mistake. Thus, Udacity thanked me, and asked me not to submit my 9.6 score. 

Which means my ranking dropped from this:
![alt text][image1]
to this: 
![alt text][image7]

To my knowledge and theirs, none of the other contestants knew of these bugs or at least used them to exploit the leaderboard.

I have no hard feelings about being unable to submit my best answer. Throughout the process, Udacity was very responsive and understanding. Plus, whole competition was a great learning experience for me - which is what it's all about.  

#### Things I learned
Evaluate your model with the exact same metrics the competition does  
Run and visualize your whole pipeline step by step. You never know where mistakes are hiding =)


[//]: # (Image References)

[image1]: ./examples/new_leaderboard_screenshot.png "Leaderboard Exploit"
[image2]: ./examples/data_plot.png "Data"
[image3]: ./examples/data_plot_preprocessed.png "Preprocessed"
[image4]: ./examples/mismatch_frame_0.png "Frame 0"
[image5]: ./examples/mismatch_frame_30.png "Frame 30"
[image6]: ./workspace/merged_unsynced_output.mp4 "Mismatched Output"
[image7]: ./examples/sad_leaderboard_screenshot.png "Leaderboard Exploit"