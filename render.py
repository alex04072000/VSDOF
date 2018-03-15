# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 20:04:30 2017

@author: Alex
"""

import re
import numpy as np
import os
import cv2

def readPFM(file):
    file = open(file, 'rb')

    header = file.readline().rstrip()
    header = header.decode('utf-8')
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width, 1)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data

def disc_blur(x, r):
    kernel = np.zeros((63, 63))
    cv2.circle(kernel, (31, 31), int(r), (1), -1)
    kernel = kernel / np.sum(kernel)
    dst = cv2.filter2D(x,-1,kernel)
    return dst
    
def rgb2lin(x):
    x = np.where(x<0.04045, x/12.92, (x/1.055+0.055/1.055)**2.4)
    return x

def lin2rgb(x):
    x = np.where(x<0.0031308, 12.92*x, 1.055*(x**(1/2.4))-0.055)
    return x

#data_path = 'Sampler\\FlyingThings3D\\'
#image_path = os.path.join(data_path, 'RGB_cleanpass\\left\\0006.png')
#disparity_path = os.path.join(data_path, 'disparity\\0006.pfm')
data_path = 'C:\\Users\\Alex\\Documents\\Alex\\Research\\VSDOF\\PBRT\\Scripts\\pavilion-night\\'
image_path = os.path.join(data_path, 'left.png')
disparity_path = os.path.join(data_path, 'disparity.pfm')

I = cv2.imread(image_path)/255.0  # image
D = readPFM(disparity_path) # disparity-map
#t = 90                      # target disparity that we want to be in focus
#m = 0.25                    # desired magnitude of the resulting shallow-depth-of-field effect
t = 746.6666815261917*0.8544643/2.9316406                      # target disparity that we want to be in focus
m = 0.15/0.8544643                    # desired magnitude of the resulting shallow-depth-of-field effect

#I = rgb2lin(I)

I_n = np.zeros((I.shape), dtype=np.float32)
I_d = np.zeros((I.shape), dtype=np.float32)
for d in np.arange(np.min(D), np.max(D), 1.0/m):
    A = np.less_equal(np.abs(D - d), 1.0/m).astype(np.float32)
    B = np.multiply(I, A)
    A_b = disc_blur(A, m * np.abs(d - t))
    B_b = disc_blur(B, m * np.abs(d - t))
    I_n = np.multiply(I_n, 1 - np.expand_dims(A_b, -1)) + B_b
    I_d = np.multiply(I_d, 1 - np.expand_dims(A_b, -1)) + np.expand_dims(A_b, -1)
I = np.divide(I_n, I_d)

#I = lin2rgb(I)

cv2.imwrite("out.png", np.round(I*255.0).astype(np.uint8))