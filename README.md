# Real-Time Emotion Detection with CNN

This is a simple project that can detect your emotions in real-time using your webcam.  
It uses a **CNN (Convolutional Neural Network)** made in **PyTorch** to guess if you’re **Angry,Happy, Sad or Neutral**.

---

## What it does

- Uses a CNN to recognise emotions
- Shows your emotions live on webcam
- Draws a box around your face and writes the emotion
- You can train it on your own dataset
- Works faster if you have a GPU

---

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

