#importing required files and modules
import numpy as np
from sklearn.cluster import KMeans
import cv2
import datetime
from matplotlib import pyplot as plt
import time

#Storing images in variable names so that they can be easily accessed later
print("Reading Images...")
image1=cv2.imread("mri1.jpg")
image2=cv2.imread("mri2.jpg")
image3=cv2.imread("Berlin.jpg")

print("Reshaping Images...")
#list of image variables
image=[image1,image2,image3]
reshaped=[0,0,0]

for i in range(0,3):
    reshaped[i] = image[i].reshape(image[i].shape[0] , image[i].shape[1], image[i].shape[2])
    
#User Inputs Value of K for image 1,2 and 3 respectively
numClusters=list(map(int,input("Enter the number of culsters for image 1,2 and 3 respectively: ").split(" ")))

"""
Main K-NN Algorithm working:

1.Choosing the number of Clusters
2.Selecting at random K points for centroid, in our case 40 was passed as the number of neighbors.
3.Assigning each Data point as we say each pixel value closest to the above centroid that further
gives us clusters.
4.Now we compute and place the new centroid for each cluster.
5.On the last step we just do the reassignment of the new nearest centroid and if in any case any
new reassignment took place we would reiterate the above process.
"""
print("Clustering...")
clustering=[0,0,0]
for i in range(0,3):
    kmeans = KMeans(n_clusters=numClusters[i], n_init=40, max_iter=500).fit(reshaped[i])
    clustering[i] = np.reshape(np.array(kmeans.labels_, dtype=np.uint8),(image[i].shape[0], image[i].shape[1]))

sortedLabels=[[],[],[]]
for i in range(0,3):
    sortedLabels[i] = sorted([n for n in range(numClusters[i])],
        key=lambda x: -np.sum(clustering[i] == x))

print("Concatinating original and segemented Images...")
kmeansImage=[0,0,0]
concatImage=[[],[],[]]
for j in range(0,3):
    kmeansImage[j] = np.zeros(image[j].shape[:2], dtype=np.uint8)
    for i, label in enumerate(sortedLabels[j]):
        kmeansImage[j][ clustering[j] == label ] = int((255) / (numClusters[j] - 1)) * i
    concatImage[j] = np.concatenate((image[j],193 * np.ones((image[j].shape[0], int(0.0625 * image[j].shape[1]), 3), dtype=np.uint8),cv2.cvtColor(kmeansImage[j], cv2.COLOR_GRAY2BGR)), axis=1)

#Displaying results
cv2.imshow('Segmented image 1',concatImage[0])
cv2.waitKey(0)
cv2.imshow('Segmented image 2',concatImage[1])
cv2.waitKey(0)
cv2.imshow('Segmented image 3',concatImage[2])
cv2.waitKey(0)

#Saving the images on computer
#File Name Format=HOUR:MINUTE:SECOND C_(number of cluster in that image).png(for Jupiter Notebook)
#File Name Format for Python file application is output 1,2 and 3
#the concatinated images will be saved in the same PWD
#this is for all three images.
fn=['output1.png','output2.png','output3.png']
print("Saving Results onto the computer...")
for i in range(0,3):
    dt = datetime.datetime.now()
    fileExtension = "png"
    filename = (str(dt.hour)
        + ':'+str(dt.minute) + ':'+str(dt.second)
        + ' C_' + str(numClusters[i]) + '.' + fileExtension)
    print(filename)
    time.sleep(1)
    cv2.imwrite(fn[i], concatImage[i])
print("Program executed")
