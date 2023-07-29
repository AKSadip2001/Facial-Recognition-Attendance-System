import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Directory containing images of individuals for attendance
path = 'ImagesAttendance'
attendance_file = f"{datetime.now().strftime('%d-%m-%Y')}.csv"

# Load images and encodings for known individuals
classNames = []
encodeListKnown = []

# Loop through each image in the specified path and load image data and corresponding encodings
for img_name in os.listdir(path):
    img_path = os.path.join(path, img_name)
    img = cv2.imread(img_path)
    if img is not None:
        classNames.append(os.path.splitext(img_name)[0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeListKnown.append(encode)

# Create or open the attendance CSV file
if not os.path.exists(attendance_file):
    # If the file doesn't exist, create it and add the header
    with open(attendance_file, 'w') as f:
        f.write('Name,Time\n')
    # Initialize the nameList as a set
    nameList = set()
else:
    # If the file exists, read the names already present in it and add them to the nameList
    with open(attendance_file, 'r') as f:
        lines = f.readlines()[1:]  # Skip the header line
        nameList = {line.split(',')[0].strip() for line in lines}

def markAttendance(name):
    # Check if the person's name is not already in the nameList
    if name not in nameList:
        # Append the person's name and time to the attendance file
        with open(attendance_file, 'a') as f:
            f.write(f'{name},{datetime.now().strftime("%H:%M:%S")}\n')
        # Add the person's name to the nameList to avoid duplicate entries
        nameList.add(name)

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        continue

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Detect and recognize faces in the current frame
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    # Keep track of recognized names in the current frame
    recognized_names = []

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        # Calculate face distance to find the best match in the known encodings
        faceDistances = face_recognition.face_distance(encodeListKnown, encodeFace)
        minDistIndex = np.argmin(faceDistances)
        minDist = faceDistances[minDistIndex]

        # Use a threshold to consider it a valid match
        if minDist < 0.5:
            name = classNames[minDistIndex].upper()
            y1, x2, y2, x1 = [pos * 4 for pos in faceLoc]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name.split()[0], (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            # Check if the person has already been recognized in this frame
            if name not in recognized_names:
                # Record attendance only for the first occurrence in the current frame
                markAttendance(name)
                recognized_names.append(name)

    cv2.imshow('Webcam', img)
    
    # Set a delay of 10 milliseconds for smoother video
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
