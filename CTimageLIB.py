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
    brainImage[brainMask] = 1
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

def connected_threshold(image, seed_point, threshold):
    # 创建一个空的分割结果图像，与输入图像大小相同
    segmentation = np.zeros_like(image, dtype=np.uint8)

    # 创建一个待处理像素集合，并将种子点添加进去
    queue = []
    queue.append(seed_point)

    # 获取输入图像的灰度值
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = image

    # 循环直到待处理像素集合为空
    while len(queue) > 0:
        # 从待处理像素集合中选择一个像素进行处理
        current_point = queue.pop(0)
        x, y = current_point

        # 检查该像素的灰度值是否在阈值范围内
        if segmentation[y, x] == 0 and abs(int(gray_image[y, x]) - int(gray_image[seed_point[1], seed_point[0]])) <= threshold:
            # 将该像素标记为分割区域的一部分，并添加到分割结果图像中
            segmentation[y, x] = 255

            # 检查该像素的4个相邻像素
            neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
            for neighbor in neighbors:
                nx, ny = neighbor
                # 如果满足条件且没有被处理过，将其添加到待处理像素集合中
                if nx >= 0 and nx < image.shape[1] and ny >= 0 and ny < image.shape[0] and segmentation[ny, nx] == 0:
                    queue.append(neighbor)

    return segmentation

def levelset_segmentation(image, iterations, time_step, lambda_val, alpha_val, epsilon_val):
    # 将图像转换为灰度图像
    gray_image = image

    # 初始化水平集函数，使用高斯函数作为初始边界
    phi = np.ones_like(gray_image, dtype=np.float32)
    phi = phi - 0.5
    phi = phi * -1.0

    # 迭代演化水平集函数
    for i in range(iterations):
        # 计算水平集函数的梯度和法向量
        grad_x = cv2.Sobel(phi, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(phi, cv2.CV_32F, 0, 1, ksize=3)
        norm = np.sqrt(grad_x ** 2 + grad_y ** 2)
        grad_x /= (norm + epsilon_val)
        grad_y /= (norm + epsilon_val)

        # 计算曲率项
        curvature = cv2.Sobel(grad_x, cv2.CV_32F, 1, 0, ksize=3) + cv2.Sobel(grad_y, cv2.CV_32F, 0, 1, ksize=3)

        # 计算速度函数
        speed = lambda_val * (np.abs(gray_image - image.mean()) - alpha_val)

        # 更新水平集函数
        phi += time_step * speed * curvature

    # 根据水平集函数的符号，计算分割结果
    result = np.uint8(phi > 0) * 255

    return result

def fastMarching(image):
    
    return False