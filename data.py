import numpy as np
import os
import cv2
import shutil
#import libraries

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #harcascade classifier

cap = cv2.VideoCapture(0) #video capture
path = "dataset/" # datapath creation
id = input('enter user name') #id creation (name of the face)


# Create target Directory and replace if new 
if os.path.exists(path+str(id)): #checking if dir is present
    shutil.rmtree(path+str(id)) #if present delete
os.mkdir(path+str(id)) #take new faces
print("Directory " , path+str(id),  " Created ")  #terminal output 
image_number=0; #instantiate no of images

while 1:
    readflag,image = cap.read()#image readflag and image
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)#grayscale conversion preprocessing
    faces = face_cascade.detectMultiScale(image, 1.3, 5)#face extractor in opencv
    for (x,y,w,h) in faces:
        image_number=image_number+1;
        cv2.imwrite(path+str(id)+ "/" +str(image_number)+ ".jpg", image[y:y+h, x:x+w])
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.waitKey(100)#delay of 100 ms it waits for keys but we wont give any
    cv2.imshow('image',image)#image on frame
    cv2.waitKey(1)#delay of 1 ms
    if image_number > 40: #41 images count
        break
cap.release()#stopping Videocapture
cv2.destroyAllWindows()#destroying all frames started by imshow
