
import cv2
import numpy as np
import os
import glob
import sys
from numpy.linalg import inv
import math 
from PIL import Image 
from scipy import ndimage


K_side=np.array([[560.3496469844627, 0.0, 797.4401137301651], [0.0, 565.2708604172439, 620.635825547462], [0.0, 0.0, 1.0]])
D_side=np.array([[-0.05812798882134729], [0.03210453823497408], [-0.03274943813682195], [0.01050921965983058]])


side_frame=cv2.imread('13_draw_skeleton_side_view/side_view.png')
tnsPoints_s = np.zeros((8, 3)) 
tnsPoints_s[ 0] = (   0.00,     0.00, 0)
tnsPoints_s[ 1] = (   262.5,   0.00, 0)
tnsPoints_s[ 2] = (   262.5,   212.5, 0)
tnsPoints_s[ 3] = (   0.00,   212.5, 0)
tnsPoints_s[ 4] = (   -5.0,   212.5, 150)
tnsPoints_s[ 5] = (   272.5,   30, 60.0)
tnsPoints_s[ 6] = (   80.0,   220.5, 150.5)
tnsPoints_s[ 7] = (   267.5,   212.5, 100)

undistorted_img = np.zeros((8, 2))
undistorted_img[ 0] =(941, 736)
undistorted_img[ 1] =(1080, 828) 
undistorted_img[ 2] =(620,816) 
undistorted_img[ 3] =(710, 728)
undistorted_img[ 4] =(712, 588)
undistorted_img[ 5] =(1043, 712)
undistorted_img[ 6] =(687, 591)
undistorted_img[ 7] =(612, 617)
 
retval, rvec, tvec  = cv2.solvePnP(tnsPoints_s,
                                   undistorted_img,
                                   np.asarray(K_side),
                                   np.asarray(D_side))
rotMat, _ = cv2.Rodrigues(rvec)

xy_start= np.array ( [[188.51188757],
 [209.87845726],
 [  50.        ]])
xy_end= np.array(  [[187.4630028],
 [185.5444073],
 [  5.       ]])
print('drawskel2-Start\n',xy_start)
print('drawskel2-end\n',xy_end)
xy_start, jac = cv2.projectPoints(xy_start, rotMat, tvec, K_side, D_side)
xy_end, jac = cv2.projectPoints(xy_end, rotMat, tvec, K_side, D_side)
xy_start=xy_start.reshape(-1,2)
xy_start=xy_start.astype('int')
xy_end=xy_end.reshape(-1,2)
xy_end=xy_end.astype('int')
cv2.line(side_frame, (xy_start[0,0],xy_start[0,1]), (xy_end[0,0],xy_end[0,1]), (0,0,255), 4,cv2.LINE_AA)
#cv2.circle(side_frame, (xy_start[0,0],xy_start[0,1]),3, (0,255,255), -1,cv2.LINE_AA)
cv2.imwrite('13_draw_skeleton_side_view/side_view_human_skeleton.png',side_frame)
