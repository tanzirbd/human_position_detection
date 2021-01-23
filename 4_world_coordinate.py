import cv2
import numpy as np
import os
import glob
import sys
from numpy.linalg import inv

DIM=(1600, 1200)
#Left
#K=np.array([[826.53409696,   0.,         800.83335041],
# [  0.,         808.2649773,  600.71940929],
 #[  0.,           0.,           1.        ]])
#D=np.array([[-0.60930172],  [0.62141923], [-0.00579851],  [0.01397509], [-0.31105334]])

#Right
K=np.array([[780.71136837,   0.,         818.29777789],
 [  0.,         795.93104327, 585.12633059],
 [  0.,           0.,           1.        ]])
D=np.array([[-0.43290854],  [0.221774],   [-0.00544102], [-0.00755772], [-0.04866457]] )

#Side
#K=np.array([[560.3496469844627, 0.0, 797.4401137301651], [0.0, 565.2708604172439, 620.635825547462], [0.0, 0.0, 1.0]])
#D=np.array([[-0.05812798882134729], [0.03210453823497408], [-0.03274943813682195], [0.01050921965983058]])



#files=np.load('stereo_calib_para.npz')
#tvec=files['tvecsR']
#rvec=files['rvecsR']
#np.asarray(rvec)
#np.asarray(tvec)

tnsPoints = np.zeros((4, 3)) 
tnsPoints[ 0] = (   0.00,     0.00, 0)
tnsPoints[ 1] = (   262.5,   0.00, 0)
tnsPoints[ 2] = (   262.3,   212.7, 0)
tnsPoints[ 3] = (   0.00,   212.5, 0)

undistorted_img = np.zeros((4, 2))

#Left    
#undistorted_img[ 0] =(463, 579)
#undistorted_img[ 1] =(1004, 555) 
#undistorted_img[ 2] =(998,984) 
#undistorted_img[ 3] =(496, 974) 

#side
#undistorted_img[ 0] =(941, 736)
#undistorted_img[ 1] =(1080, 828) 
#undistorted_img[ 2] =(620,816) 
#undistorted_img[ 3] =(710, 728) 

#Right
distorted_img[ 0] =(880,910)
distorted_img[ 1] =(410, 860)
distorted_img[ 2] =(376, 490) 
distorted_img[ 3] =(880, 461)

retval, rvec, tvec  = cv2.solvePnP(tnsPoints,
                                   distorted_img,
                                   np.asarray(K),
                                   np.asarray(D))
#print(rvec)
axis=np.float32([[262.4,0,0],[0,212.5,0],[0,0,238.4]])
rotMat, _ = cv2.Rodrigues(rvec)
#imgpt,_jac=cv2.projectPoints(axis,rvec,tvec,K,D)
#print(imgpt)


def groundProjectPoint(image_point, z = 0.0):
    camMat = np.asarray(K)
    iRot = inv(rotMat)
    iCam = inv(camMat)
    uvPoint = np.ones((3, 1))

    # Image point
    uvPoint[0, 0] = image_point[0]
    uvPoint[1, 0] = image_point[1]

    tempMat = np.matmul(np.matmul(iRot, iCam), uvPoint)
    tempMat2 = np.matmul(iRot, tvec)
    #print(tempMat)
    s = (z + tempMat2[2, 0]) / tempMat[2, 0]
    wcPoint = np.matmul(iRot, (np.matmul(s * iCam, uvPoint) - tvec))
    # wcPoint[2] will not be exactly equal to z, but very close to it
    assert int(abs(wcPoint[2] - z) * (10 ** 8)) == 0
    wcPoint[2] = z
    
    #Left
    #wcPoint[0]+=22
    #wcPoint[1]-=7
    
    #Right 
    #wcPoint[0]+=22
    #wcPoint[1]-=7    
    return wcPoint

pixel =(424, 608) 
print("Pixel:" ,(pixel))
print('World Coordinate X and Y in cm')
xy=groundProjectPoint(pixel)
print(xy.shape)
print('X= ',xy[0])
print('Y= ',xy[1])
