from matplotlib import pyplot as plt
import os
import numpy as np
import pydicom
import cv2

# Function of read CT image using the pydicom
def readCT(dataPath):
    dataset = pydicom.dcmread(dataPath)
    return dataset

# Threshold method of segmentation
def thresholdSegmentation(pixels,minThreshold,maxThreshold):
    # check the data validity
    if minThreshold >maxThreshold:
        raise ValueError("minThreshold should be smaller than maxThreshold.")
    # judge by the threshold
    brainMask = (pixels > minThreshold) & (pixels < maxThreshold)
    brainImage = pixels.copy()
    brainImage[~brainMask] = 0
    return brainImage

def regionGrowing(image, seed_point, threshold):

    # the region growing algorithm
    # image: the input image
    # seed_point: the original point that start to grow
    # threshold: threshold growing

    # initialize the original image
    segmented_image = np.zeros_like(image, dtype=np.uint8)
    
    # calculate the scale of the image
    height, width = image.shape
    
    # create the matrix of visited pixel
    visited = np.zeros((height, width), dtype=np.uint8)
    
    # growing queue
    queue = []
    queue.append(seed_point)
    
    # get the grey value of pixels
    seed_value = image[seed_point[1], seed_point[0]]
    
    # define adjacent position of pixels
    neighbors = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    
    # Region grwoing iteration
    while len(queue) > 0:
        # get the point from the queue
        current_point = queue.pop(0)
        
        # check the pixel was visited or not
        if visited[current_point[1], current_point[0]] == 1:
            continue
        
        # visit current pixel
        visited[current_point[1], current_point[0]] = 1
        
        # check the difference of current pixel and the seed point
        current_value = image[current_point[1], current_point[0]]
        diff = abs(int(current_value) - int(seed_value))
        
        # judge whether meet the growing condition
        if diff <= threshold:
            # set current pixel as the part of the result
            segmented_image[current_point[1], current_point[0]] = current_value
            
            # add the adjacent pixels to the queue
            for neighbor in neighbors:
                x = current_point[0] + neighbor[0]
                y = current_point[1] + neighbor[1]
                
                # check the adjacent pixel condition of inside the image
                if x >= 0 and x < width and y >= 0 and y < height:
                    # add the unvisited pixels to the queue
                    if visited[y, x] == 0:
                        queue.append((x, y))
    
    return segmented_image