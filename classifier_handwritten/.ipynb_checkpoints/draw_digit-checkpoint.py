# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 10:36:28 2021

@author: HP
"""

import cv2
import numpy as np
import math #to compute euclidian distance
import torch
import argparse

# Construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("--digit", required=True,
   help="digit you plan to draw")
ap.add_argument("--number", required=True,
   help="the number of the digit image (ex: number=1 means that it's the first image")
ap.add_argument("--background_color", required=True, type=str, 
   help="background color (black or white)")
args = vars(ap.parse_args())

#### Python function for drawing with the finger


#start the recording
cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height

#Test if the camera opened correctly
if not (cap.isOpened()):
    raise IOError("Cannot open webcam")

#open a template image of a finger (directed to the ceiling)
finger_template = cv2.imread('finger_template2.jpg', 0)
w, h = finger_template.shape[::-1] #we get the shape of the finger_template
method = eval('cv2.TM_CCOEFF_NORMED') #method of template matching, see https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html


#Create a counter to count time
count_motionless = 0

#initialize the values returned by template matching method (to measure difference)
min_val, max_val, min_loc, max_loc = (0,0),(0,0), (0,0), (0,0)

#set a value of epsilon : distance (in pixels) within which we consider 2 points being at the same location
epsilon = 3 #value in pixels !
 
#Create a black image on which we'll draw
draw = np.zeros((480,640), np.uint8)

#have a count for a test image: when the program runs we might not want to draw directly
#since we're not ready, so we wait for the drawing of one image to be ready
count_test=0




while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Convert the frame into a gray image
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #flip the image (to not have a symmetry)
    gray_img = cv2.flip(gray_img, 1)
    if args['background_color']=='white':
        #invert the image
        gray_img = (255-gray_img)
    
    #Apply the template matching method
    res = cv2.matchTemplate(gray_img,finger_template,method)
    min_val_next, max_val_next, min_loc_next, max_loc_next = cv2.minMaxLoc(res) #max_loc contains the coordinate of the top_left 
    
    #If the values of maximum are located approximatively at the same location
    dist_w = abs(max_loc_next[0] - max_loc[0]) #distance in width between the current location of the finger and the previous one
    dist_h = abs(max_loc_next[1] - max_loc[1]) #distance in height between the current location of the finger and the previous one
    #print the distance (if not zero)
    if (dist_w!=0) or (dist_h!=0):
        print("dist with math.dist : ", math.dist(max_loc_next, max_loc))
        print("dist with pixels : ", dist_w + dist_h)
        
    #we deal with the case where the detection would bug : dist very high, we want to draw if the location is consistent
    #we also only draw when there is a motion (dist is greater than epsilon)
    if (dist_w+dist_h>= epsilon) and (dist_w+dist_h<=15):
        #draw a circle at the center of the top phalanx of the finger
        cv2.circle(draw, (max_loc_next[0] + w//2, max_loc_next[1] +h//2), 8, 255,-1)
        count_motionless=0 #count for a motionless finger
        
    if (dist_w+dist_h<=epsilon): #if the finger don't move, it's motionless, we increase the count
        count_motionless+=1
     
    ####Print text####
    text = np.zeros((480,640), np.uint8)  #new image where there will be the text   
    if count_test==0:
        cv2.putText(text, "TEST DRAWING, BE READY FOR THE NEXT DRAWING" ,
                    (30,100),cv2.FONT_HERSHEY_SIMPLEX , 1, (255,255,255),2,cv2.LINE_AA)

    #if there is a motion :
    if count_motionless<=30 :
        cv2.putText(text, "Drawing..." ,(30,400),
                    cv2.FONT_HERSHEY_SIMPLEX , 1, (255,255,255),2,cv2.LINE_AA)
    
    #if it's been a long time since there is no motion
    if count_motionless > 30:
        cv2.putText(text, "Drawing in pause" ,(30,400),
                    cv2.FONT_HERSHEY_SIMPLEX , 1, (255,255,255),2,cv2.LINE_AA)
        
    #Put the count (normalized to a count over 10)
    cv2.putText(text, "Count : " ,(30,460),
                cv2.FONT_HERSHEY_SIMPLEX , 1, (255,255,255),2,cv2.LINE_AA)    
    cv2.putText(text, str(int(count_motionless/10))+"/10" ,(150,460),
                cv2.FONT_HERSHEY_SIMPLEX , 1, (255,255,255),2,cv2.LINE_AA)
        
    #if it's been some time, we save the image
    if count_motionless==100 :
        file_name = 'data/'+str(args['digit'])+'_'+str(args['number'])+'.jpg'
        #save the image
        cv2.imwrite(file_name, draw)
        if count_test==1:#this means that it's the real drawing so we stop
            break
        count_test+=1 #now we are ready for the real drawing
        #reinitialize for next drawing
        draw = np.zeros((480,640), np.uint8) #resinitialize the drawing image
        count_motionless=0
        
    # Display the resulting frame
    cv2.imshow('frame', gray_img)
    # Display the drawing
    cv2.imshow('drawing',draw+text)
    
    #update values for the next iteration
    min_val, max_val, min_loc, max_loc = min_val_next, max_val_next, min_loc_next, max_loc_next
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()