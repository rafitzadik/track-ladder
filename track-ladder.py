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

def init_net():
    caffe.set_mode_gpu()
    print 'start net init'
    net = caffe.Net('/home/rafi/test_fcn/ladder-fcn/ladder-fcn32s-opt/deploy.prototxt', 
                    '/home/rafi/test_fcn/ladder-fcn/ladder-fcn32s-opt/snapshot/train_iter_16000_keep.caffemodel', caffe.TEST)
                
    print 'finished net init'
    return net

def init_realsense():
    pyrs.start()
    dev = pyrs.Device()
    return dev

def find_ladder(image, depth, net, rs_dev):
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
    
#    cv2.circle(image, my_cnt_center, 5, (0,0,255), 2)
    
    mask = np.ones(depth.shape, np.uint8) #this is for a numpy mask; start with a mask that masks everything
    cv2.drawContours(mask, [my_cnt], 0, 0, -1) #unmask the area that is identified by the neural net
    #but re-mask everything except the middle 50% in height:
    bbox = cv2.boundingRect(my_cnt) #bbox = (x,y,w,h)
    mask2 = np.ones(depth.shape, np.uint8) #another mask that masks everything
    cv2.rectangle(mask2, (bbox[0]+bbox[2]/4,bbox[1]+bbox[3]/4), (bbox[0]+bbox[2]*3/4, bbox[1]+bbox[3]*3/4), 0, -1)
    mask = cv2.bitwise_or(mask, mask2)
    
    dpt_masked = np.ma.array(depth, mask=mask)
    dpt_masked = np.ma.masked_outside(dpt_masked, 1000, 10000) #further mask values below 1m or above 10m
    #Option 0 - find the average... not a great option    
    #likely_depth = np.ma.average(dpt_masked)
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
        
    center_3d = rs_dev.deproject_pixel_to_point(np.array([my_cnt_center[0], my_cnt_center[1]], np.uint), likely_depth)
    
    #print average, likely_depth, center_3d
    
    #let's see where the likely depth actually is in the picture:
    thresh = threshold(depth, threshmin = likely_depth-300, threshmax = likely_depth+500, newval = 0)
    thresh = threshold(thresh, threshmax = 1, newval = 255) #make remaining values 255
    thresh = thresh.astype(np.uint8) 
    cnt_mask = np.zeros(thresh.shape, dtype=np.uint8)
    cv2.drawContours(cnt_mask, [my_cnt], 0, 255, -1)    

    mask3 = np.bitwise_and(thresh, cnt_mask)
    pink = np.zeros(image.shape, dtype=np.uint8)
    cv2.rectangle(pink, (0,0), pink.shape[0:2], (200, 0, 200), -1)
    pink_mask = cv2.bitwise_and(pink, pink, mask=mask3)
    image = cv2.addWeighted(pink_mask, 0.5, image, 0.5, 0)
    #Here we just draw on the image to show
    # Start by drawing a grid at height -100
    for x in range(-1500, 1700, 200):
        start = rs_dev.project_point_to_pixel(np.array([x, 500, 1000]))
        end = rs_dev.project_point_to_pixel(np.array([x, 500, 7000]))
        if center_3d[0] > x and center_3d[0] <= x + 200:
            cv2.line(image, tuple(start), tuple(end), (0,255,255), 1)
        else:
            cv2.line(image, tuple(start), tuple(end), (0,0,255), 1)
    for z in range(1000, 7000, 200):
        start = rs_dev.project_point_to_pixel(np.array([-1500, 500, z]))
        end = rs_dev.project_point_to_pixel(np.array([1500, 500, z]))
        if center_3d[2] > z and center_3d[2] <= z + 200:
            cv2.line(image, tuple(start), tuple(end), (0,255,255), 1)
        else:
            cv2.line(image, tuple(start), tuple(end), (0,0,255), 1)
    
    cv2.drawContours(image, [my_cnt], 0, (255,0,0), 3)
    #print origin, go_z, go_xz
    cv2.putText(image, '({:.0f}, {:.0f}, {:.0f})'.format(center_3d[0], center_3d[1], center_3d[2]),(0,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
    return image, center_3d

#main:
if len(sys.argv) < 3:
    print "usage: {} <output_file> <image_1> [<image_2>, ...]".format(sys.argv[0])
    print "   NOTE: expect to find the depth png for img-XXXX.jpg at dpt-XXXX.png"
    exit(-1)

out_txt = open(sys.argv[1], 'w')
out_vid = None

net = init_net()
rs_dev = init_realsense()

for im_name in sys.argv[2 :]:
    dpt_name = im_name[:-12]+'dpt-'+im_name[-8:-4]+'.png'    
    print 'analyze image ', im_name, ' depth ', dpt_name
    image = cv2.imread(im_name)
    depth = cv2.imread(dpt_name, 2) #2 says to preserve the 16 bits
    if image is None or depth is None:
        print "couldn't open image or depth, skipping"
    else:
        image, center_3d = find_ladder(image, depth, net, rs_dev)
        cv2.imshow('image', image)
        out_txt.write('{}: ({:.0f}, {:.0f}, {:.0f})\n'.format(im_name, center_3d[0], center_3d[1], center_3d[2]))
        if out_vid is None:
            try:
                out_vid = cv2.VideoWriter(sys.argv[1][:-4]+'.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (image.shape[1], image.shape[0]), True)
            except:
                print "problem opening output stream"
                exit(1)
            if not out_vid.isOpened():
                print "output stream not open"
                exit(1)
        out_vid.write(image)
        k = cv2.waitKey(1) & 0xff
        if k == ord('q'):
            break

out_vid.release()     
 