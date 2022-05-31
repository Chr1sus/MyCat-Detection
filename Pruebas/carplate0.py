import cv2 
from matplotlib import pyplot as plt
import numpy as np
#import imutils

lic_data=cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')
image=cv2.imread('/home/chrisus/carplate/audi.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


number = lic_data.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors=5,minSize=(50,50))
 
print("number plate detected:" +str(len(number))) 
for numbers in number:
    (x,y,w,h)=numbers
    roi_gray=gray[y:y+h, x:x+w]
    roi_color=image[y:y+h, x:x+h]
    cv2.rectangle(image,(x,y),(x+w,y+h),(0, 255, 0),3)
 
cv2.imwrite("carplate.jpg", image)
print('Se ejecuto carplate.py')





