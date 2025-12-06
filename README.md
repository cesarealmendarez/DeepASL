# DeepASL

![deep_asl_ex_vid_AdobeCreativeCloudExpress](https://user-images.githubusercontent.com/89669770/160533990-ae71afe4-67f4-4d21-93b4-8bf65fba739f.gif)

## Overview

DeepASL uses a webcam video feed and Python to interpret American Sign Language (ASL) hand gestures in real time. The goal of this project was to learn the fundamentals of [Convolutional Neural Networks](https://cs231n.github.io/convolutional-networks/) and to understand how [Computer Vision](http://vision.stanford.edu/teaching/cs131_fall2122/index.html) can make them interactively useful in real-world applications.

## Installation and Usage

1. Clone the repository:  
   ```git clone https://github.com/cesarealmendarez/DeepASL.git```

2. Navigate to the project directory:  
   ```cd DeepASL```

3. Install the required packages:  
   ```pip3 install opencv-python mediapipe numpy```

4. Run DeepASL:  
   ```python3 app.py```

## What’s on My Screen?

Once you run DeepASL, two windows will appear:

- **Analytics Window:** Displays the raw video feed along with extracted data points used to interpret hand landmarks, steadiness, depth perception, output confidence, and snapshot triggering.
- **Hand Segmentation Window:** Shows how the network breaks the image down into a pattern of 1s and 0s, prompting its best attempt to guess which ASL letter you are showing.

## Resources Used

1. [MediaPipe](https://github.com/google/mediapipe): Used to detect the shape of the hand and create a skeleton-like outline for segmenting useful classification features.
2. [MNIST Handwritten Digits Classification using a Convolutional Neural Network (CNN)](https://towardsdatascience.com/mnist-handwritten-digits-classification-using-a-convolutional-neural-network-cnn-af5fafbc35e9) 
3. [A Comprehensive Guide to Convolutional Neural Networks — the ELI5 Way](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53) 
4. [Simple Introduction to Convolutional Neural Networks](https://towardsdatascience.com/simple-introduction-to-convolutional-neural-networks-cdf8d3077bac)
