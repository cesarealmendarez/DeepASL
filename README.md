# DeepASL

![deep_asl_ex_vid_AdobeCreativeCloudExpress](https://user-images.githubusercontent.com/89669770/160533990-ae71afe4-67f4-4d21-93b4-8bf65fba739f.gif)

**Overview**

DeepASL utilizes webcam video feed and some Python code to interpret American Sign Language hand gestures in real time! My goal with this project was to learn the very basics of what makes up [Convolutional Neural Networks](https://cs231n.github.io/convolutional-networks/) and how [Computer Vision](http://vision.stanford.edu/teaching/cs131_fall2122/index.html) can make them interactively useful in the real world.

**Installation and Usage**

1. Clone this repo: ```git clone https://github.com/cesarealmendarez/DeepASL.git```
2. Navigate to project: ```cd DeepASL```
3. Install required packages: ```pip3 install opencv-python mediapipe numpy```
4. Run DeepASL: ```python3 app.py```

**What's on my Screen?**

Once you run DeepASL, two windows will appear, the Analytics window displays the raw video along with the extracted data points used in interpreting hand landmarks/steadiness, depth perception, output confidence, and finally triggering the snapshot. The Hand Segmentation window shows what the network will break down into a pattern of 1's and 0's prompting it's best attempt to guess what ASL letter you're showing!

**Resources Used**

1. [MediaPipe](https://github.com/google/mediapipe): Used to perceive the shape of the hand and create a skeleton-like outline, enabling segmentation of useful classification features.
2. [MNIST Handwritten Digits Classification using a Convolutional Neural Network (CNN)](https://towardsdatascience.com/mnist-handwritten-digits-classification-using-a-convolutional-neural-network-cnn-af5fafbc35e9) 
3. [A Comprehensive Guide to Convolutional Neural Networks — the ELI5 way](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53) 
4. [Simple Introduction to Convolutional Neural Networks](https://towardsdatascience.com/simple-introduction-to-convolutional-neural-networks-cdf8d3077bac)