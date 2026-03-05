#imports
import cv2 # for video capture and face detection
import torch 

from torchvision import transforms
import time
from PIL import Image
from model import EmotionCNN #cnn model

#Load trained model
model_cnn =EmotionCNN()
model_cnn.load_state_dict(torch.load("emotion_cnn.pth"))
model_cnn.eval()

#Emotion label
emotions_labels =["Angry", "Happy","Sad","Neutral"]

#image transform for webcam input
transform_image = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48,48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

#Load face detector, keeps your face as the focus of the webcam feed
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
