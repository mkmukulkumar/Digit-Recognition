import cv2
import os
import numpy as np
path='TrainingData'
myList=os.listdir(path)
images=[]
classNo=[]
#entering into directory and opening every image in sub directory
#Then appending it to images and mapping its identity in classNo array
for x in range(0,len(myList)):
    myPicList=os.listdir(path+"/"+str(x))
    for y in myPicList:
        curImg=cv2.imread(path+"/"+str(x)+"/"+y,cv2.IMREAD_GRAYSCALE)
        curImg=cv2.resize(curImg,(40,40))
        curImg=curImg.flatten()
        images.append(curImg)
        classNo.append(x)
images=np.array(images,dtype=np.float32)
classNo=np.array(classNo)

#kNN train
knn=cv2.ml.KNearest_create()
knn.train(images,cv2.ml.ROW_SAMPLE,classNo)

##Testing
testing=[]
myList=os.listdir('TestingData')
for x in myList:    
    test=cv2.imread('TestingData/'+str(x),cv2.IMREAD_GRAYSCALE)   
    test=cv2.resize(test,(40,40))
    test=test.flatten()
    testing.append(test)
testing=np.array(testing,dtype=np.float32)
ret, result, neighbours, dist=knn.findNearest(testing,k=3)

print(result)