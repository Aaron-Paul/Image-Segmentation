# Image Segmentation
## Abstract:
Image Segmentation is the process by which a digital image is partitioned into various subgroups (of pixels) called Image Objects.
The goal of segmentation is to simplify and/or change the representation of an image into something that is more meaningful and easier to analyze.Image segmentation is typically used to locate objects and boundaries in images. 

## Folder Structure and their Uses:
1.Generic Object Segmentation:This folder contains a python file that will convert the given images into segmented images with the help of K-Means Clustering Algorithm.The Images used to test this program can be found in the Output Folder.<br>
2.Medical Image Enhancement :This folder contains a python program as well as a jupyter notebook file that will enchance a image by segmentation and also increasing the contrast with the help of the KNN algorithm <br>
3.Lane Detection:Lane detection plays a key role in the vision-based driver assistance system and is used for vehicle navigation, lateral control, collision prevention, or lane departure warning system. We present an adaptive method for detecting lane marking based on the intensity of road images in night scene which is the cause of numerous accidents.<br>
4.Road Segmentation using Sattelite Images:The src folder contains the source code for the program.The data folder contain link to the images used for training the model and the model folder contains the final model program.The necessary requirement needed to run this program is given in the requirement.txt file <br>
5.Output:this file contains the images used for testing the above programs and the results obtained by these programs

## Group Members and Their contribution:
- Aaron:Generic Object Segmentation and Medical Image Segmentation
- Pushpak:Lane Segmentation Program
- Pradhi and Aditya:Road Segmentation using Sattelite Images

## Steps to run the included code:
- Generic Object Segmentation:
  - Command line Arguments:python imageSegmentation.py K inputImageFilename outputImageFilename  (where K is greater than 2.)
- Medical Image Enhancement:
  - Install Anaconda and run the jupyter notebook with the images to be segemented in the same working directory as of the notebook and simply follow through with the code execution.
- Lane Segmentation Program:
  - Set the path of the video file of road to be segmented in the main function of the program and execute the program.
- Road Segmentation using Sattelite Images
  - Install jupyter notebook and take it to the PWD of.ipynb with the images already there and simply follow through  with the code execution.You could even aquire our own sattelite images with the aquire_data.py file and run the program after changing the file paths in the program files.
## Algorithm
- Generic Object Segmentation:
The K-means algorithm is an iterative technique that is used to partition an image into K clusters.The basic algorithm is<br>
1.Pick K cluster centers, either randomly or based on some heuristic method.<br>
2.Assign each pixel in the image to the cluster that minimizes the distance between the pixel and the cluster center<br>
3.Re-compute the cluster centers by averaging all of the pixels in the cluster<br>
4.Repeat steps 2 and 3 until convergence is attained (i.e. no pixels change clusters)<br>
In this case, distance is the squared or absolute difference between a pixel and a cluster center. The difference is typically based on pixel color, intensity, texture, and location, or a weighted combination of these factors.
K can be selected manually, randomly, or by a heuristic. This algorithm is guaranteed to converge, but it may not return the optimal solution. The quality of the solution depends on the initial set of clusters and the value of K.
- Medical Image Enhancement:
The steps Involved in K-NN algorithm:<br>
1.Choosing the number of Clusters<br>
2.Selecting at random K points for centroid, in our case 40 was passed as the number of neighbors.<br>
3.Assigning each Data point as we say each pixel value closest to the above centroid that further gives us clusters.<br>
4.Now we compute and place the new centroid for each cluster.<br>
5.On the last step we just do the reassignment of the new nearest centroid and if in any case any new reassignment took place we would reiterate the above process.<br>
- Lane Segmentation Program:
First, a region of interest (ROI) image is extracted from the original image and
converted to its grayscale image. After that, we segment out regions likely to be lane
using a novel idea based on lane width. Finally, based on the structural and
orientation properties of a lane only those segments are kept which match with these
required properties. Once lane segments are obtained, hough transform can be applied
to estimate lane boundaries. Cognitive methods are then applied to combine results of
previous frame with the current one to avoid any miscalculation.
- Road Segmentation using Sattelite Images
To solve the road segmentation problem, we used an Unet, it is a fully convolutional network, with 3 cross-connections. Adam optimiser with a learning rate of 0.00001 was used, along with dice loss (because of the unbalanced nature of the dataset.) The model trained for 61 epochs before earlstopper kicked in and killed the training process. A validation dice loss of 0.7548 was achieved.<br>
Filename: acquire_data.py<br>
Function:  Downloads the Massachusetts Roads Dataset or the Massachusetts Buildings Dataset. <br>
Filename: build_dataset.py<br>
Function: Crops the images into small 256x256 images and divides the dataset into training and testing set.Splits the dataset into train and test sets and stores then in an ImageDataAugmentation friendly format. <br>
Filename: get_data_generators.py<br>
Function: Builds and returns data generators.<br>
Filename: loss_functions.py<br>
Function: To deal with the imbalanced data issue<br>
Filename: Road_Detection_GPU.ipynb (Jupiter Notebook)<br>
Function: Building the model and executing it to see the predicted output of road segmentation.<br>

## Uses of Image Segmentation:
- Improved Quality of MRI/Medical images for better detection of diseases and problems
- Segmenting images can help to improve robot vision
- Image segementation is also be applied to satellite images in order to get better object     detection.These objects may be buildings,roads,cars,or trees , for example. Applications of this type of imagery labelling are widespread, from analysing traffic to monitoring environmental changes taking place due to global warming.
## Results
### 1.Generic Object Segmentation Program:
![Burma](output/burma.jpg)
![Burma](output/burma-segmented.jpg)

![Duomo](output/duomo.jpg)
![Duoma](output/duomo-segmented.jpg)

![tunnel](output/tunnel.png)
![tunnel](output/tunnel-segmented.png)

![face](output/myface.jpeg){:height="200px" width="100px"}
![face](output/output.jpeg){:height="200px" width="100px"}

### 2.Medical Image Enhancement Program:

![Results](output/Application_Result_1.png) <br>
![Results](output/Application_Result_2.png) <br>
![Results](output/Application_Result_3.png)

### 3.Lane Detection Program:
![Lane](output/results.png)

### 4.Satellite Image Segmentation Program:
![RoadDetection](output/Program_Output.png)

