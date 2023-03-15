import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path='ImagesAttendence'
images=[]
classNames=[]
myList=os.listdir(path)
print(myList)

for cl in myList:
    curImg=cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

print(classNames)

# functions to compute all the encodings
def findEncodings(images):
    encodeList=[]
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendence(name):
    with open('Attendence.csv','r+') as f:
        myDataList=f.readlines()
        nameList=[]
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            now=datetime.now()
            dtString=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

        # print(myDataList)

# markAttendence('elon')

encodeListKnown=findEncodings(images)
print('Encoding completes')

cap=cv2.VideoCapture(0)

while True:
    success,img=cap.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)

    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis=face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)
        matchIndex=np.argmin(faceDis)

        if matches[matchIndex]:
            name=classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1=faceLoc
            y1, x2, y2, x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendence(name)

    cv2.imshow('webcam',img)
    cv2.waitKey(1)

# # for getting the face and converting it from bgr to rgb   STEP1
# imgElon=face_recognition.load_image_file('ImagesBasic/elon.jpg')
# imgElon=cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
#
# # for getting the face and converting it from bgr to rgb
# imgTest=face_recognition.load_image_file('ImagesBasic/elonTest.jpg')
# imgTest=cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)
#
# # for finding the face and its location for imgElon STEP 2
# faceLoc=face_recognition.face_locations(imgElon)[0]
# encodeElon=face_recognition.face_encodings(imgElon)[0]
# cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
#
# # for finding the face and its location for imgTest
# faceLocTest=face_recognition.face_locations(imgTest)[0]
# # here we are getting encodings for the faces of the person to be tested
# encodeTest=face_recognition.face_encodings(imgTest)[0]
# cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)
#
# # Compare the two face and finding the distance between them
# # here we compare the two encodings to recognise the face
# # here we use linear svm to find that they match or not
# results=face_recognition.compare_faces([encodeElon],encodeTest)
#
#
# # when there are a lot of images and we have to find out how much similar these images are or to find the best match
# faceDis=face_recognition.face_distance([encodeElon],encodeTest)
# print(results,faceDis)
# cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
