from PIL import Image
from keras.applications.vgg16 import preprocess_input
import base64
from io import BytesIO
import json
import random
import cv2
from keras.models import load_model
import numpy as np

from keras.preprocessing import image
model = load_model('C:\\Users\\Admin\\Desktop\\Face Recognition\\facefeatures_new_model.h5')

# Loading the cascades
face_cascade = cv2.CascadeClassifier('C:\\Users\\Admin\\Desktop\\Face Recognition\\haarcascade_frontalface_default.xml')

class_labels = ['Cong Kha','Danh Hieu','Danh Hoa' ]

video_capture = cv2.VideoCapture(0)

# Doing some Face Recognition with the webcam

while True:
    _, frame = video_capture.read()
    #canvas = detect(gray, frame)
    #image, face =face_detector(frame)
    
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    if faces is ():
        face = None
    else:
    # Crop all faces found
        for (x,y,w,h) in faces:
            label_position = (x,y)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,255),2)
            face = frame[y:y+h, x:x+w]

    if type(face) is np.ndarray:
        face = cv2.resize(face, (200, 200))

        im = Image.fromarray(face, 'RGB')
           #Resizing into 200x200 because we trained the model with this image size.
        img_array = np.array(im)
                    #Our keras model used a 4D tensor, (images x height x width x channel)
                    #So changing dimension 200x200x3 into 1x200x200x3 
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)[0]
        label = class_labels[pred.argmax()]
        print(pred)
        print(label)
        cv2.putText(frame,label,label_position, cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
    else:
        cv2.putText(frame,"No face found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
