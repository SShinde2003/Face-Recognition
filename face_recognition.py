import cv2
print(dir(cv2.face))


import cv2 as cv
import numpy as np
from PIL import Image
import os

path="D:\Projects\Final Year\dataset" #location of your dataset

recognizer=cv.face.LBPHFaceRecognizer_create()
detector=cv.CascadeClassifier("file:///C:/Users/LENOVO/Desktop/face/LBPH-Face-Recognizer-main/haarcascade_frontalface_default.xml")#location of the haar cascade xml file

def getImageAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    ids = []
    for imagePath in imagePaths:
        try:
            # Extract the ID from the filename
            file_name = os.path.split(imagePath)[-1]
            parts = file_name.split("_")
            if len(parts) > 1:  # Ensure there are enough parts after splitting
                id = int(parts[1])
            else:
                print(f"Skipping invalid file format: {file_name}")
                continue
            
            # Load the image and convert to grayscale
            pilImage = Image.open(imagePath).convert('L')
            imageNp = np.array(pilImage, 'uint8')
            
            # Append the data
            faces.append(imageNp)
            ids.append(id)
        except Exception as e:
            print(f"Error processing file {imagePath}: {e}")
    return faces, ids



print("\n Training faces")
faces,ids=getImageAndLabels(path)

recognizer.train(faces,np.array(ids))

recognizer.write("E:\LBPH-Face-Recognizer-main/trainer.yml")

print("Model is trained on "+str(len(np.unique(ids)))+" no of faces")
