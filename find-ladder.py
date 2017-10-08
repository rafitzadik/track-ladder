#!/usr/bin/python

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 19:36:46 2017

@author: rafi
"""

import sys
import cv2
import numpy as np
from scipy.stats import threshold

import _init_paths
import caffe

import pyrealsense as pyrs

min_cnt_moment = 50 #min mass of contour to be considered

if len(sys.argv) != 3:
    print "usage: {} <color image> <depth png>".format(sys.argv[0])
    exit(-1)

pyrs.start()
dev = pyrs.Device()

image = cv2.imread(sys.argv[1])
depth = cv2.imread(sys.argv[2], 2) #2 says to preserve the 16 bits

#init FCN
caffe.set_mode_gpu()
print 'start net init'
net = caffe.Net('/home/rafi/test_fcn/ladder-fcn/ladder-fcn32s-opt/deploy.prototxt', 
                '/home/rafi/test_fcn/ladder-fcn/ladder-fcn32s-opt/snapshot/train_iter_16000_keep.caffemodel', caffe.TEST)
            
print 'finished net init'

#find ladder
in_ = np.array(image, dtype=np.float32)
#in_ = in_[:,:,::-1] #using cv2.imread, we are already BGR
in_ -= np.array((104.00698793,116.66876762,122.67891434))
in_ = in_.transpose((2,0,1))
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_
net.forward()
out = net.blobs['score'].data[0].argmax(axis=0)

#keep only 'ladder' class
pre_thresh = np.empty_like(out, dtype=np.uint8)
np.copyto(pre_thresh, out, casting='unsafe')
thresh = threshold(pre_thresh, threshmin = 1, threshmax = 1, newval = 0) #throw out all values that are too small or too large
thresh = threshold(thresh, threshmax = 1, newval = 255) #make remaining values 255
thresh = thresh.astype(np.uint8) 

_, contours, _ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnt_area = [cv2.contourArea(cnt) for cnt in contours]
my_cnt = contours[np.argmax(cnt_area)]
M = cv2.moments(my_cnt)
my_cnt_center = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))

cv2.drawContours(image, [my_cnt], 0, (255,0,0), 3)
cv2.circle(image, my_cnt_center, 5, (0,0,255), 2)

mask = np.ones(depth.shape, np.uint8) #this is for a numpy mask; start with a mask that masks everything
cv2.drawContours(mask, [my_cnt], 0, 0, -1) #unmask the area that is identified by the neural net
#but re-mask everything except the middle 50% in height:
bbox = cv2.boundingRect(my_cnt) #bbox = (x,y,w,h)
mask2 = np.ones(depth.shape, np.uint8) #another mask that masks everything
cv2.rectangle(mask2, (bbox[0]+bbox[2]/4,bbox[1]+bbox[3]/4), (bbox[0]+bbox[2]*3/4, bbox[1]+bbox[3]*3/4), 0, -1)
mask = cv2.bitwise_or(mask, mask2)

dpt_masked = np.ma.array(depth, mask=mask)
dpt_masked = np.ma.masked_outside(dpt_masked, 1000, 10000) #further mask values below 1m or above 10m
average = np.ma.average(dpt_masked)
depth_hist, bin_edges = np.histogram(np.ma.compressed(dpt_masked), 90, (1000, 10000)) #calculate the histogram of values between 70cm and 10m
#Option 1 to find the likely depth: the histogram bin with the most values
#likely_depth = bin_edges[np.argmax(depth_hist)] #the bin with the most values
#Option 2: the smallest (closest) bin with meaningful values
likely_depth = -1
for i in range(len(depth_hist)):
    if depth_hist[i] > 100:
        likely_depth = bin_edges[i]
        break
if likely_depth < 0:
    likely_depth = bin_edges[np.argmax(depth_hist)] #nothing has more than 100?? Just use the bin with the most values
    
center_3d = dev.deproject_pixel_to_point(np.array([my_cnt_center[0], my_cnt_center[1]], np.uint), likely_depth)

print average, likely_depth, center_3d

#let's see where the likely depth actually is in the picture:
thresh = threshold(depth, threshmin = likely_depth-300, threshmax = likely_depth+500, newval = 0)
thresh = threshold(thresh, threshmax = 1, newval = 255) #make remaining values 255
thresh = thresh.astype(np.uint8) 
cnt_mask = np.zeros(thresh.shape, dtype=np.uint8)
cv2.drawContours(cnt_mask, [my_cnt], 0, 255, -1)

mask3 = np.bitwise_and(thresh, cnt_mask)
im_masked = cv2.bitwise_and(image, image, mask=mask3)
cv2.imshow('image', image)
cv2.imshow('im_masked', im_masked)
mask2 = mask2*255
cv2.imshow('mask', mask2)
cv2.waitKey(0)