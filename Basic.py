import cv2
import numpy as np
import face_recognition

# for getting the face and converting it from bgr to rgb   STEP1
imgElon=face_recognition.load_image_file('ImagesBasic/elon.jpg')
imgElon=cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)

# for getting the face and converting it from bgr to rgb
imgTest=face_recognition.load_image_file('ImagesBasic/elonTest.jpg')
imgTest=cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

# for finding the face and its location for imgElon STEP 2
faceLoc=face_recognition.face_locations(imgElon)[0]
encodeElon=face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

# for finding the face and its location for imgTest
faceLocTest=face_recognition.face_locations(imgTest)[0]
# here we are getting encodings for the faces of the person to be tested
encodeTest=face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

# Compare the two face and finding the distance between them
# here we compare the two encodings to recognise the face
# here we use linear svm to find that they match or not
results=face_recognition.compare_faces([encodeElon],encodeTest)


# when there are a lot of images and we have to find out how much similar these images are or to find the best match
faceDis=face_recognition.face_distance([encodeElon],encodeTest)
print(results,faceDis)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Elon Musk',imgElon)
cv2.imshow('Elon Test',imgTest)
cv2.waitKey(0)