# Real-Time Emotion Classifier

A small PyTorch convolutional network that watches your webcam, finds your face and
guesses which of four emotions you are showing. I built it to understand how CNNs
actually work, not to ship a usable emotion tool.

## What it does

`train.py` trains the network on a folder of labelled face images and writes the
weights to `emotion_cnn.pth`. `live runner.py` then opens the webcam, locates faces
with OpenCV's Haar cascade detector, crops each one and runs it through the network.
The predicted label is drawn above a green box on the live feed. Press q to quit.

## The model

It is deliberately tiny. Two convolutional blocks (1 to 32 to 64 channels, 3x3 kernels,
ReLU and 2x2 max pooling after each), then a flatten and two fully connected layers
down to four scores. Input is a 48x48 greyscale crop normalised to the range -1 to 1.
Training is Adam at a learning rate of 0.001, cross entropy loss, batch size 64, ten epochs.

The webcam path applies exactly the same transforms as the training path. Getting that
wrong is the classic way to end up with a model that scores well on paper and predicts
nonsense on real input, so both live in the same shape of `transforms.Compose`.

## Running it

Python 3.8 or newer, plus:

```bash
pip install torch torchvision opencv-python Pillow
```

The dataset is not in the repo. `train.py` expects `data/train` and `data/test`, each
holding one subfolder per class named `Angry`, `Happy`, `Sad` and `Neutral`. Any image
size works, the loader resizes to 48x48.

```bash
python train.py
python "live runner.py"
```

The quotes matter, the filename has a space in it.

## Known limitations

- The class order is wrong. `ImageFolder` numbers the folders alphabetically (Angry,
  Happy, Neutral, Sad) but `live runner.py` lists them as Angry, Happy, Sad, Neutral,
  so those last two labels come out swapped at inference time.
- `train.py` builds a test loader and never uses it. The accuracy printed each epoch is
  training accuracy, so there is no honest held-out number for this model.
- Haar cascades want a well-lit, roughly front-on face. Side angles are usually missed.

## Why I made it

I made this project to **learn how CNNs work** and see if I can make a program that reads emotions.  
It’s a fun way to play around with computer vision and PyTorch.
