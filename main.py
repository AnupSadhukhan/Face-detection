import cv2
import sys
import urllib
import numpy as np

#Loading image from url
req = urllib.urlopen('https://www.hellomagazine.com/imagenes/healthandbeauty/skincare-and-fragrances/2019102679678/simon-cowell-face-new-youthful-appearance/0-384-786/simon-cowell-t.jpg')
arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
#loading training data
cascPath = "C:\Python27\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imdecode(arr, -1)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.05,
    minNeighbors=5,
    minSize=(30, 30)
)

print "Yaah!! {0} face detected...".format(len(faces))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Face Detected", image)
cv2.waitKey(0)
