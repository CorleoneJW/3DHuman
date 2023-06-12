from matplotlib import pyplot as plt
import os
import numpy as np
import pydicom

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