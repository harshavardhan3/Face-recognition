import cv2
import imutils.paths as paths
import face_recognition
import pickle
import os
#import libraries

dataset = "dataset/"# path of the data set 
file = "encodings/encoding1.pickle" # path of the pickle file 

imagepaths = list(paths.list_images(dataset))#list of all the image paths
Encodings = []#encoding (face recognition lib gives 128 dim vector for each face)
Names = []#names of the faces

for (i, imagePath) in enumerate(imagepaths): #for the vector number and image path in total paths
    print("Processing Image {}/{}".format(i + 1,len(imagepaths)))
    name = imagePath.split(os.path.sep)[-2]#gives name from folder name
    image = cv2.imread(imagePath)#image read
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)#preprocessing to rgb	
    boxes = face_recognition.face_locations(rgb, model= "hog")#box having face locations
    encodings = face_recognition.face_encodings(rgb, boxes)#encoding given by face recognition encoding
    for encoding in encodings:
       Encodings.append(encoding)
       Names.append(name)
       print("encodings...")
       data = {"encodings": Encodings, "names": Names}
       output = open(file, "wb") 
       pickle.dump(data, output)
       output.close()
