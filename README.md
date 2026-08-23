# Real-Time Emotion Classifier

A small PyTorch convolutional network that watches your webcam, finds your face and
guesses which of four emotions you are showing. I built it to understand how CNNs
actually work, not to ship a usable emotion tool.

## What it does

`train.py` trains the network on a folder of labelled face images and writes the
weights to `emotion_cnn.pth`. `live runner.py` then opens the webcam, locates faces
with OpenCV's Haar cascade detector, crops each one and runs it through the network.
The predicted label is drawn above a green box on the live feed. Press q to quit.

## Files

- `model.py` – The CNN model  
- `dataset.py` – Loads and processes the images for training/testing  
- `train.py` – Trains the CNN and saves the model  
- `live_runner.py` – Opens your webcam and predicts emotions live  

---

## What you need

Python 3.8+ and these packages:

```bash
pip install torch torchvision opencv-python Pillow
```

---

## Dataset

Put your images in this way:

```
data/
  train/
    Angry/
    Happy/
    Sad/
    Neutral/
  test/
    Angry/
    Happy/
    Sad/
    Neutral/
```

Images should be **48x48**, but the code will resize them if not.

---

## Training

Run this to train your model:

```bash
python train.py
```

- It will show loss and accuracy for each epoch  
- Saves the model as `emotion_cnn.pth`

---

## Using the Webcam

first make sure your webcam is connected and on!

After training, run:

```bash
python live_runner.py
```

- Webcam opens and detects your face  
- Shows predicted emotion on screen  
- Press `q` to quit  

---

## How it works

1. Detects your face using OpenCV  
2. Converts the face to greyscale and resizes it  
3. CNN predicts your emotion  
4. Draws a rectangle and emotion text on the video  

---
## Clone this git Repo

```bash
git clone https://github.com/Nishchay-Bhudia/Emotions-Classifier-CNN.git
```
---


## Why I made it

I made this project to **learn how CNNs work** and see if I can make a program that reads emotions.  
It’s a fun way to play around with computer vision and PyTorch.
