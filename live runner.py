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

#start webcam feed
capture = cv2.VideoCapture(0)

while True:
                                     # captures each frame of the webcam feed
    ret, frame = capture.read()
    if not ret:
        break

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    #makes it grey just like the training data

    faces = face_cascade.detectMultiScale(grey, 1.3, 5)    # detects face

    for (x, y, w, h) in faces:

        face = grey[y:y+h, x:x+w]
        face_pil = Image.fromarray(face)

        img = transform_image(face_pil).unsqueeze(0)

        with torch.no_grad():
            output = model_cnn(img)
            _, predicted = torch.max(output, 1)        #makes emotions predictions 

        emotion = emotions_labels[predicted.item()]

        #draw rectangle around face on the webcam
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

        #draw text - the classified emotions
        cv2.putText(
            frame,
            emotion,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

    cv2.imshow("Emotion Detector", frame)   #display the frame

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break   #exit 

capture.release()
cv2.destroyAllWindows()