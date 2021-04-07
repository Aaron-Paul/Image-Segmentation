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

#reshaped list contains attributes of image and will be used to later to assign the clusters colour

for i in range(0,3):
    reshaped[i] = image[i].reshape(image[i].shape[0] , image[i].shape[1], image[i].shape[2])
    
#User Inputs Value of clusters for image 1,2 and 3 respectively
    
numClusters=list(map(int,input("Enter the number of clusters for image 1,2 and 3 respectively: ").split(" ")))

"""
Main Algorithm working:

1.Choosing the number of Clusters(n)
2.kmeans centers points for centroid.
3.Assigning each Data point as we say each pixel value closest to the above centroid that further
gives us clusters.
4.in our case 41 was passed as the number of neighbors
5.On the last step we just do the concatination of original and segmented image
"""
print("Clustering...")
#segmented images classes attributes
clustering=[0,0,0]

#calculating 3 colour of clusters using built-in k means algorithm
for i in range(0,3):
    kmeans = KMeans(n_clusters=numClusters[i], n_init=41, max_iter=500).fit(reshaped[i])
    clustering[i] = np.reshape(np.array(kmeans.labels_, dtype=np.uint8),(image[i].shape[0], image[i].shape[1]))

#assigning a data point to one of the clusters
sortedLabels=[[],[],[]]
for i in range(0,3):
    sortedLabels[i] = sorted([n for n in range(numClusters[i])],key=lambda x: -np.sum(clustering[i] == x))

print("Concatinating original and segemented Images...")

#Concatinating original and segemented Images
#first the orginal image is appended to KNNImage then with each iteration
#the assigned data points in sortedLabel are appended to concatenated image.
#concatimage contains the final result (axis=1,means the concatination is happening hortizontally) and we use the BGR scale for better contrast.

KNNImage=[0,0,0]
concatImage=[[],[],[]]
for j in range(0,3):
    KNNImage[j] = np.zeros(image[j].shape[:2], dtype=np.uint8)
    for i, label in enumerate(sortedLabels[j]):
        KNNImage[j][clustering[j] == label] = int((255) / (numClusters[j] - 1)) * i
    concatImage[j] = np.concatenate((image[j],193 * np.ones((image[j].shape[0], int(0.0625 * image[j].shape[1]), 3), dtype=np.uint8),cv2.cvtColor(KNNImage[j], cv2.COLOR_GRAY2BGR)), axis=1)

#Displaying results
cv2.imshow('Segmented image 1',concatImage[0])
cv2.waitKey(0)
cv2.imshow('Segmented image 2',concatImage[1])
cv2.waitKey(0)
cv2.imshow('Segmented image 3',concatImage[2])
cv2.waitKey(0)

#Saving the images on computer
#File Name Format=HOUR:MINUTE:SECOND C_(number of cluster in that image).png(for Jupiter Notebook)
#File Name Format for Python file application is output 1,2 and 3 (for Python Application)
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
