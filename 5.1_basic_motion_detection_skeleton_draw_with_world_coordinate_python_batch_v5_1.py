import cv2
import numpy as np
import os
import glob
import sys
from numpy.linalg import inv
import math 
from PIL import Image 
from scipy import ndimage
#import xlsxwriter 
from argparse import ArgumentParser
import json

from modules.input_reader import VideoReader, ImageReader
from modules.draw import Plotter3d, draw_poses
from modules.parse_poses import parse_poses


import xlwt 
from xlwt import Workbook 
#left
#K=np.array([[283.85661626,   0.,         695.08737514],
# [  0.,         284.17161762, 627.64761443],
# [  0.,           0. ,          1.        ]]
#)
#D=np.array([[-0.02343873],
# [-0.04107063],
# [ 0.02405401],
# [-0.00561532]]
#)

#right
K=np.array([[780.71136837,   0.,         818.29777789],
 [  0.,         795.93104327, 585.12633059],
 [  0.,           0.,           1.        ]])
D=np.array([[-0.43290854],  [0.221774],   [-0.00544102], [-0.00755772], [-0.04866457]] )

#img = cv2.imread('/home/tanzir/Downloads/WorkPlace/20_05_08_newdata/chessboard_two/left/80.png')
#imgL = cv2.imread('left_rectify.png')
#imgR = cv2.imread('right_rectify.png')
frame_top=cv2.imread('cap_skel/image3time-0.2.png')
#frame1= cv2.line(img, (600,700), (650,700), (0,0,255), 4,cv2.LINE_AA)
#frame1= cv2.line(frame1, (650,700), (650,550), (0,0,255), 4,cv2.LINE_AA)
#cv2.imwrite('test.png',frame1)  
q = np.array([[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00, -9.54693871e+02],
 [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00, -6.00547028e+02],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  8.02098010e+02],
 [ 0.00000000e+00,  0.00000000e+00,  2.17212763e-01, -0.00000000e+00]])
#Disparity settings


def disp(points):
    pts_src = np.array([[498, 974], [463, 580], [790, 706],[910, 916],[225, 454],[358, 1084],[600,1135],[1088,963],[1116,463],[1133,1155],[176,877],[1107,673],[1117,784]])# [797, 822][1143,1172]#[585,326],[468,416],[1436,506],[1155,1061]
    # corresponding points from image 2 (i.e. (154, 174) matches (212, 80))
    pts_dst = np.array([[710, 728],[940, 736],[906, 782],[732, 800] ,[957,540],[714,582],[682,634],[572,861],[982,890],[615,553],[827,530],[994,883],[848,892]])#[824,781],[616,527]#[989,564],[1018,990],[1001,516],[476,672]
 
    # calculate matrix H
    h, status = cv2.findHomography(pts_src, pts_dst)
 
    # provide a point you wish to map from image 1 to image 2
    #a = np.array([[154, 174]], dtype='float32')
    a =np.array(points)
    a=np.array(a,dtype='float32')
    a=a.reshape(1,1,2)
    print(a.shape)
    # finally, get the mapping
    pointsOut = cv2.perspectiveTransform(a, h)
    print(pointsOut) 
    return pointsOut





#cx=K[0,2]
#cy=K[1,2]
#cap= cv2.VideoCapture(1, cv2.CAP_V4L)
#cap.set(cv2.CAP_PROP_FPS, 15)

#cap.set(cv2.CAP_PROP_FRAME_WIDTH,1600)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1200)

def distance(x2,y2,x3,y3,x4,y4,x5,y5):
    dist1 = int(math.sqrt((x2 - cx)**2 + (y2 - cy)**2))
    dist2 = int(math.sqrt((x3 - cx)**2 + (y3 - cy)**2))  
    #print(dist)  
    dist3 = int(math.sqrt((x4 - cx)**2 + (y4 - cy)**2))
    dist4 = int(math.sqrt((x5 - cx)**2 + (y5 - cy)**2))
    #print(dist1,dist2,dist3,dist4)  
    #print(dist1)
    if (dist1<dist2) and (dist1<dist3) and (dist1<dist4):
       return (x2,y2)
    elif (dist2<dist1) and (dist2<dist3) and (dist2<dist4):
       return (x3,y3)
    elif (dist3<dist1) and (dist3<dist2) and (dist3<dist4): 
       return (x4,y4)
    elif (dist4<dist1) and (dist4<dist2) and (dist4<dist3): 
       return (x5,y5)
   
#fourcc = cv2.VideoWriter_fourcc('X','V','I','D')

#out = cv2.VideoWriter("output.avi", fourcc, 5.0, (640,480))


def groundProjectPoint(image_point, z = 0.0):
    tnsPoints = np.zeros((4, 3)) 
    tnsPoints[ 0] = (   0.00,     0.00, 0)
    tnsPoints[ 1] = (   262.3,   0.00, 0)
    tnsPoints[ 2] = (   262.5,   212.5, 0)
    tnsPoints[ 3] = (   0.00,   212.7, 0)

    distorted_img = np.zeros((4, 2))    
#Left
#undistorted_img[ 0] =(876, 900)
#undistorted_img[ 1] =(408, 848) 
#undistorted_img[ 2] =(376,476) 
#undistorted_img[ 3] =(882, 450)

#Right 
    distorted_img[ 0] =(880,910)
    distorted_img[ 1] =(410, 860) 
    distorted_img[ 2] =(376, 490) 
    distorted_img[ 3] =(880, 461)  

    retval, rvec, tvec  = cv2.solvePnP(tnsPoints,
                                   distorted_img,
                                   np.asarray(K),
                                   np.asarray(D))

    rotMat, _ = cv2.Rodrigues(rvec)
    camMat = np.asarray(K)
    iRot = inv(rotMat)
    iCam = inv(camMat)
    uvPoint = np.ones((3, 1))

    # Image point
    uvPoint[0, 0] = image_point[0]
    uvPoint[1, 0] = image_point[1]

    tempMat = np.matmul(np.matmul(iRot, iCam), uvPoint)
    tempMat2 = np.matmul(iRot, tvec)

    s = (z + tempMat2[2, 0]) / tempMat[2, 0]
    wcPoint = np.matmul(iRot, (np.matmul(s * iCam, uvPoint) - tvec))
    #s=s+8
    # wcPoint[2] will not be exactly equal to z, but very close to it
    assert int(abs(wcPoint[2] - z) * (10 ** 8)) == 0
    wcPoint[2] = z
    
    #Left
    #wcPoint[0]+=5
    #wcPoint[1]-=6

    #Right
    wcPoint[0]+=5
    #wcPoint[1]-=2
    return wcPoint


#print(img_center)
def rotate_poses(poses_3d, R, t):
    R_inv = np.linalg.inv(R)
    for pose_id in range(len(poses_3d)):
        pose_3d = poses_3d[pose_id].reshape((-1, 4)).transpose()
        pose_3d[0:3, :] = np.dot(R_inv, pose_3d[0:3, :] - t)
        poses_3d[pose_id] = pose_3d.transpose().reshape(-1)

    return poses_3d


def humanSkeleton(images,img_st,img_end,img_circle,poses_3d,frame1,x_dr,y_dr,min_pix_dis):
    parser = ArgumentParser(description='Lightweight 3D human pose estimation demo. '
                                        'Press esc to exit, "p" to (un)pause video or process next image.')
    parser.add_argument('-m', '--model',
                        help='Required. Path to checkpoint with a trained model '
                             '(or an .xml file in case of OpenVINO inference).',
                        type=str, required=True)
    parser.add_argument('--video', help='Optional. Path to video file or camera id.', type=str, default='')
    parser.add_argument('-d', '--device',
                        help='Optional. Specify the target device to infer on: CPU or GPU. '
                             'The demo will look for a suitable plugin for device specified '
                              '(by default, it is GPU).',
                        type=str, default='GPU')
    parser.add_argument('--use-openvino',
                        help='Optional. Run network with OpenVINO as inference engine. '
                             'CPU, GPU, FPGA, HDDL or MYRIAD devices are supported.',
                        action='store_true')
    parser.add_argument('--images', help='Optional. Path to input image(s).', nargs='+', default=images)
    parser.add_argument('--height-size', help='Optional. Network input layer height size.', type=int, default=256)
    parser.add_argument('--extrinsics-path',
                        help='Optional. Path to file with camera extrinsics.',
                        type=str, default=None)
    parser.add_argument('--fx', type=np.float32, default=-1, help='Optional. Camera focal length.')
    args = parser.parse_args()

   # if args.video == '' and args.images == '':
    #    raise ValueError('Either --video or --image has to be provided')

    stride = 8
    if args.use_openvino:
        from modules.inference_engine_openvino import InferenceEngineOpenVINO
        net = InferenceEngineOpenVINO(args.model, args.device)
    else:
        from modules.inference_engine_pytorch import InferenceEnginePyTorch
        net = InferenceEnginePyTorch(args.model, args.device)

    #canvas_3d = np.zeros((720, 640, 3), dtype=np.uint8)
    #plotter = Plotter3d(canvas_3d.shape[:2])
    #canvas_3d_window_name = 'Canvas 3D'
    #cv2.namedWindow(canvas_3d_window_name)
    #cv2.setMouseCallback(canvas_3d_window_name, Plotter3d.mouse_callback)

    file_path = args.extrinsics_path
    if file_path is None:
        file_path = os.path.join('data', 'extrinsics.json')
    with open(file_path, 'r') as f:
        extrinsics = json.load(f)
    R = np.array(extrinsics['R'], dtype=np.float32)
    t = np.array(extrinsics['t'], dtype=np.float32)

    frame_provider = args.images
    is_video = False
    if args.video != '':
        frame_provider = VideoReader(args.video)
        is_video = True
    base_height = args.height_size
    fx = args.fx

    delay = 1
    esc_code = 27
    p_code = 112
    space_code = 32
    mean_time = 0
    count=0
    frame=frame_provider
    if True:
        current_time = cv2.getTickCount()
        #if frame is None:
            #break
        input_scale = base_height / frame.shape[0]
        scaled_img = cv2.resize(frame, dsize=None, fx=input_scale, fy=input_scale)
        scaled_img = scaled_img[:, 0:scaled_img.shape[1] - (scaled_img.shape[1] % stride)]  # better to pad, but cut out for demo
        if fx < 0:  # Focal length is unknown
            fx = np.float32(0.8 * frame.shape[1])

        inference_result = net.infer(scaled_img)
        poses_3d, poses_2d = parse_poses(inference_result, input_scale, stride, fx, is_video)
        edges = []
        if len(poses_3d):
            poses_3d = rotate_poses(poses_3d, R, t)
            poses_3d_copy = poses_3d.copy()
            #print(poses_3d_copy.shape)
            x = poses_3d_copy[:, 0::4]
            y = poses_3d_copy[:, 1::4]
            z = poses_3d_copy[:, 2::4]
            #print(np.average(z))

  
            #print(poses_3d_copy[:, 2::4][0,0])
            poses_3d[:, 0::4], poses_3d[:, 1::4], poses_3d[:, 2::4] = -z, x, -y
            X=poses_3d[:, 0::4]
            Y=poses_3d[:, 1::4]
            Z=poses_3d[:, 2::4]
            Xavg=np.average(X)
            Yavg=np.average(Y)
            Zmax=np.amax(Z)
            Zmin=np.amin(Z)
            print(Zmin)
            print(Zmax)
            wb = Workbook() 
  
            # add_sheet is used to create sheet. 
            sheet1 = wb.add_sheet('sheet_1')

            sheet1.write(0, 0, 'World Coordinate')
            sheet1.write(1, 2, 'measure the value in m')
            sheet1.write(2, 0, 'From model data')
            sheet1.write(2, 7, 'before add World Center coordinate')
            sheet1.write(2, 4, 'Global World Center coordinate')
            sheet1.write(2, 12, 'After add World Center coordinate')
            sheet1.write(3, 0, 'X')
            sheet1.write(3, 1, 'Y')
            sheet1.write(3, 2, 'Z')
            sheet1.write(3, 4, 'X')
            sheet1.write(3, 5, 'Y')
            sheet1.write(3, 7, 'X')
            sheet1.write(3, 8, 'Y')
            sheet1.write(3, 9, 'Z') 
            sheet1.write(3, 12, 'X')
            sheet1.write(3, 13, 'Y')
            sheet1.write(3, 14, 'Z')     
            img_st=[]
            img_end=[]
            img_circle=[]
            _,_,img_st,img_end,img_circle=draw_poses(frame, poses_2d,img_st,img_end,img_circle)
            print(min_pix_dis)
            x_y_world=groundProjectPoint(min_pix_dis)
            print(x_y_world)
            sheet1.write(4, 4, float(x_y_world[0]/100))
            sheet1.write(4, 5, float(x_y_world[1]/100))
            for a in range(0,19):
                x_mod=float(X[0,a]/100)
                y_mod=float(Y[0,a]/100)
                z_mod=float(Z[0,a]/100)
                x_mod2=float((X[0,a]-Xavg)/100)
                y_mod2=float((Y[0,a]-Yavg)/100)
                z_mod2=float((Z[0,a]-Zmin)/100)
                
                sheet1.write((a+4), 0, x_mod)
                sheet1.write((a+4), 1, y_mod)
                sheet1.write((a+4), 2, z_mod) 
                sheet1.write((a+4), 7, x_mod2)
                sheet1.write((a+4), 8, y_mod2)
                sheet1.write((a+4), 9, z_mod2) 
            for a in range(0,19):
                X[0,a]=((X[0,a]-Xavg)+x_y_world[0])
                Y[0,a]=((Y[0,a]-Yavg)+x_y_world[1])
                Z[0,a]=(Z[0,a]-Zmin)
                x_mod3=float(X[0,a]/100)
                y_mod3=float(Y[0,a]/100)
                z_mod3=float(Z[0,a]/100)
                sheet1.write((a+4), 12, x_mod3)
                sheet1.write((a+4), 13, y_mod3)
                sheet1.write((a+4), 14, z_mod3) 

            poses_3d[:, 0::4], poses_3d[:, 1::4], poses_3d[:, 2::4] = X, Y, Z  
            #print('3d World Coordinate\n \tX\t\tY\tZ\n',poses_3d)   
            poses_3d = poses_3d.reshape(poses_3d.shape[0], 19, -1)[:, :, 0:3]
            #print('check2',poses_3d.shape)
            edges = (Plotter3d.SKELETON_EDGES + 19 * np.arange(poses_3d.shape[0]).reshape((-1, 1, 1))).reshape((-1, 2))
            print('3d World Coordinate in cm\n \tX\t\tY\tZ\n',poses_3d)
            #np.savez('3d_data/image'+m+'_data_ID-'+str(i)+'_time-'+time+'.npz',poses_3d=poses_3d)
            wb.save(path3+'image'+m+'_data_ID-'+str(i)+'_time-'+time+'.xls')
       # plotter.plot(canvas_3d, poses_3d, edges)
        #cv2.imshow(canvas_3d_window_name, canvas_3d)
        #if i==1:
        #cv2.imwrite('3d_skel/image'+m+'_3d_ID-'+str(i)+'_time-'+time+'.png', canvas_3d)
        #else:
         #  cv2.imwrite('3d_skel_2/image'+m+'_3d_ID-'+str(i)+'_time-'+time+'.png', canvas_3d)
        current_time = (cv2.getTickCount() - current_time) / cv2.getTickFrequency()
        if mean_time == 0:
            mean_time = current_time
        else:
            mean_time = mean_time * 0.95 + current_time * 0.05
        #cv2.putText(frame, 'FPS: {}'.format(int(1 / mean_time * 10) / 10),
                    #(40, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
        #cv2.imshow('ICV 3D Human Pose Estimation', frame)
        #cv2.imwrite('2d_skel/image'+m+'_2d_ID-'+str(i)+'_time-'+time+'.png', frame)
        key = cv2.waitKey(delay)
        #if key == esc_code:
            #break
        #if key == p_code:
         #   if delay == 1:
          #      delay = 0
           # else:
           #     delay = 1
        #if delay == 0 or not is_video:  # allow to rotate 3D canvas while on pause
         #   key = 0
          #  while (key != p_code
                  # and key != esc_code
                 #  and key != space_code):
                #plotter.plot(canvas_3d, poses_3d, edges)
               # cv2.imshow(canvas_3d_window_name, canvas_3d)
              #  key = cv2.waitKey(3)
           #if key == esc_code:
                #break
            #if key != esc_code:
             #   delay = 1
        count+=1  
    return (img_st,img_end,img_circle,poses_3d)

def drawSkeleton1(frame1,x,y,img_st,img_end,img_circle,frame_top):
     y=y-(crop_img.shape[1])
     for i in range(len(img_st)):
         #print('Origin',x,y)
         cv2.line(frame1, ((x-img_st[i][1]),(y+(img_st[i][0]))), ((x-img_end[i][1]),(y+(img_end[i][0]))), (0,0,255), 4,cv2.LINE_AA)
         cv2.line(frame_top, ((x-img_st[i][1]),(y+(img_st[i][0]))), ((x-img_end[i][1]),(y+(img_end[i][0]))), (0,0,255), 4,cv2.LINE_AA)
         xy_pos1=((x-img_st[i][1]),(y+(img_st[i][0])))
         xy_pos2=((x-img_end[i][1]),(y+(img_end[i][0])))
         xy_start=disp(xy_pos1)
         xy_end=disp(xy_pos2)
         xy_start=xy_start.reshape(-1,2)
         xy_start=xy_start.astype('int')
         xy_end=xy_end.reshape(-1,2)
         xy_end=xy_end.astype('int')
         #print(xy_start.shape)
         #cv2.line(side_frame, (xy_start[0,0],xy_start[0,1]), (xy_end[0,0],xy_end[0,1]), (0,0,255), 4,cv2.LINE_AA)
        
     #print('Origin',x,y)
     for i in range(len(img_circle)):
        cv2.circle(frame1, (x-(img_circle[i][1]),(y+img_circle[i][0])),3, (0,255,255), -1,cv2.LINE_AA)    
        cv2.circle(frame_top, (x-(img_circle[i][1]),(y+img_circle[i][0])),3, (0,255,255), -1,cv2.LINE_AA)
        x_y=(x-(img_circle[i][1]),(y+img_circle[i][0]))
        #print(x_y)
        x_x=disp(x_y)
        x_x=x_x.reshape(-1,2)
        x_x=x_x.astype('int')
        #print(x_x[0,0])
        #print(x_x[0,1])
        #cv2.circle(side_frame, (x_x[0,0],x_x[0,1]),3, (0,255,255), -1,cv2.LINE_AA)
     #cv2.imwrite('side_view_skel/sideframe_'+m+'.png',side_frame)
     cv2.imwrite(path1+'image'+m+'time-'+time+'.png',frame1)
     cv2.imwrite(path2+'image'+m+'time-'+time+'.png',frame_top)
     return (frame1,frame_top)
def drawSkeleton2(frame1,x,y,img_st,img_end,img_circle,frame_top):
     
    for i in range(len(img_st)):
        cv2.line(frame1, ((x-(img_st[i][0])),(y-(img_st[i][1]))), ((x-(img_end[i][0])),(y-(img_end[i][1]))), (0,0,255), 4,cv2.LINE_AA)
        cv2.line(frame_top, ((x-(img_st[i][0])),(y-(img_st[i][1]))), ((x-(img_end[i][0])),(y-(img_end[i][1]))), (0,0,255), 4,cv2.LINE_AA)
        xy_pos1=(x-(img_st[i][0])),(y-(img_st[i ][1]))
        xy_pos2=(x-(img_end[i][0])),(y-(img_end[i][1]))
        xy_start_world=groundProjectPoint(xy_pos1)
        xy_end_world=groundProjectPoint(xy_pos2)        
        xy_start=disp(xy_pos1)
        xy_end=disp(xy_pos2)
        xy_start=xy_start.reshape(-1,2)
        xy_start=xy_start.astype('int')
        xy_end=xy_end.reshape(-1,2)
        xy_end=xy_end.astype('int')
        #print(xy_start.shape)
        #cv2.line(side_frame, (xy_start[0,0],xy_start[0,1]), (xy_end[0,0],xy_end[0,1]), (0,0,255), 4,cv2.LINE_AA) 
       
    for i in range(len(img_circle)):
        cv2.circle(frame1, ((x-(img_circle[i][0])),(y-(img_circle[i][1]))),3, (0,255,255), -1,cv2.LINE_AA)
        cv2.circle(frame_top, ((x-(img_circle[i][0])),(y-(img_circle[i][1]))),3, (0,255,255), -1,cv2.LINE_AA)
        x_y=((x-(img_circle[i][0])),(y-(img_circle[i][1])))
        x_y_world=groundProjectPoint(x_y) 
        #print(x_y)
        x_x=disp(x_y)
        x_x=x_x.reshape(-1,2)
        x_x=x_x.astype('int')
        #print(x_x[0,0])
        #print(x_x[0,1])
        #cv2.circle(side_frame, (x_x[0,0],x_x[0,1]),3, (0,255,255), -1,cv2.LINE_AA)
        #print (x_x)
    cv2.imwrite(path1+'image'+m+'time-'+time+'.png',frame1)
    cv2.imwrite(path2+'image'+m+'time-'+time+'.png',frame_top)
    return (frame1,frame_top)

def drawSkeleton4(frame1,x,y,img_st,img_end,img_circle,frame_top):
    #print('Origin Pixel',x,y)
    #y=y-(crop_img.shape[1])
    x=x-(crop_img.shape[0])
    for i in range(len(img_st)):
        cv2.line(frame1, ((x+img_st[i][1]),y-img_st[i][0]), ((x+img_end[i][1]),y-img_end[i][0]), (0,0,255), 4,cv2.LINE_AA)
        cv2.line(frame_top, ((x+img_st[i][1]),y-img_st[i][0]), ((x+img_end[i][1]),y-img_end[i][0]), (0,0,255), 4,cv2.LINE_AA)
        xy_pos1=((x+img_st[i][1]),y-img_st[i][0])
        xy_pos2=((x+img_end[i][1]),y-img_end[i][0])
        xy_world_start=groundProjectPoint(xy_pos1)
        xy_world_end=groundProjectPoint(xy_pos2)
        xy_start=disp(xy_pos1)
        xy_end=disp(xy_pos2)
        xy_start=xy_start.reshape(-1,2)
        xy_start=xy_start.astype('int')
        xy_end=xy_end.reshape(-1,2)
        xy_end=xy_end.astype('int')
        print(xy_start.shape)
        #cv2.line(side_frame, (xy_start[0,0],xy_start[0,1]), (xy_end[0,0],xy_end[0,1]), (0,0,255), 4,cv2.LINE_AA) 
        
    for i in range(len(img_circle)):
         cv2.circle(frame1, ((x+img_circle[i][1]),y-img_circle[i][0]),3, (0,255,255), -1,cv2.LINE_AA)
         cv2.circle(frame_top, ((x+img_circle[i][1]),y-img_circle[i][0]),3, (0,255,255), -1,cv2.LINE_AA)
         x_y=((x+img_circle[i][1]),y-img_circle[i][0])
         print(x_y)
         x_x=disp(x_y)
         x_x=x_x.reshape(-1,2)
         x_x=x_x.astype('int')

         #cv2.circle(side_frame, (x_x[0,0],x_x[0,1]),3, (0,255,255), -1,cv2.LINE_AA)
    cv2.imwrite(path1+'image'+m+'time-'+time+'.png',frame1)
    cv2.imwrite(path2+'image'+m+'time-'+time+'.png',frame_top)
    return (frame1,frame_top)

###################
## START FROM HERE
###################


time=12.1

for j in range(122,202):
    m=str(j)
    w=j+1
    n=str(w)
    time2=time+0.1
    time2=round(time2,2)
    time=str(time) 
    time2=str(time2)
#SET INPUT PATH
    path='11_capture_original_image/cap_skel/'
    frame1=cv2.imread(path+'image'+m+'time-'+time+'.png')
    frame2=cv2.imread(path+'image'+n+'time-'+time2+'.png')
    frame_top=cv2.imread(path+'image3time-0.2.png')
    img_width =frame1.shape[1]
    img_height=frame1.shape[0]
    img_center=(int(img_width/2),int(img_height/1.7))
    cx=img_center[0]
    cy=img_center[1]
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    i=1
    #OUTPUT PATH SET
    path1='12_top_view_human_skel/top_view_skel/'
    path2='13_no_human_only_skel_topview/top_view_only_skel/'
    path3='14_old_data_excel/old_data/'
    cv2.imwrite(path1+'image'+m+'time-'+time+'.png',frame1)
    cv2.imwrite(path2+'image'+m+'time-'+time+'.png',frame_top)
    wb = Workbook()
    sheet1 = wb.add_sheet('sheet_1')
    wb.save(path3+'image'+m+'_data_ID-'+str(i)+'_time-'+time+'.xls')
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        M = cv2.moments(contour)
        #print(M)
        c_x = int(M["m10"] / M["m00"])
        c_y = int(M["m01"] / M["m00"])
        
        if cv2.contourArea(contour) < 10000:
            continue
        
    #cv2.imshow("undistorted", undistorted_img)
        #cv2.imwrite("frameun_left.jpg", undistorted_img)
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        x1=x+w
        y1=y+h
        x_mid=int((x+x1)/2)
        y_mid=int((y+y1)/2)
        #y1_mid=int((y1+x1)/2)
        #y_mid=int((y+y1)/2)
        #print(x,y,x+w,y+h)
        #cv2.circle(frame1, (cx, cy), 1, (0, 0, 255), 2)
        min_pix_dis=distance(x_mid,y,x,y_mid,x1,y_mid,x_mid,y1)
        #print(min_pix_dis)
        crop_img = frame1[y:y+h, x:x+w]
        print(crop_img.shape)
        x_dr=crop_img.shape[1]+x
        y_dr=crop_img.shape[0]+y
        img_st=[]
        img_end=[]
        img_circle=[]
        poses_3d=[]
        #print(x,y)
        if min_pix_dis[0]>img_center[0] & min_pix_dis[1]<img_center[1] :
           crop_img = ndimage.rotate(crop_img, 90)
           mark=1
           cv2.imwrite('cropped_image_ID-'+str(i)+'.png',crop_img)
           img_st,img_end,img_circle,poses_3d=humanSkeleton(crop_img,img_st,img_end,img_circle,poses_3d,frame1,x_dr,y_dr,min_pix_dis)
           frame1,frame_top=drawSkeleton1(frame1,x_dr,y_dr,img_st,img_end,img_circle,frame_top)            
           cv2.circle(frame1, (min_pix_dis[0],min_pix_dis[1]), 2, (0,0,255),4)
           print('1')
        elif min_pix_dis[0]>img_center[0] & min_pix_dis[1]>img_center[1] :
           crop_img = ndimage.rotate(crop_img, 180)
           mark=2 
           cv2.imwrite('cropped_image_ID-'+str(i)+'.png',crop_img)
           img_st,img_end,img_circle,poses_3d=humanSkeleton(crop_img,img_st,img_end,img_circle,poses_3d,frame1,x_dr,y_dr,min_pix_dis)
           #print(img_st[1][1])                     
           print('2') 
           frame1,side_frame=drawSkeleton2(frame1,x_dr,y_dr,img_st,img_end,img_circle,frame_top)           
           #cv2.imwrite('cropped_image_ID-'+str(i)+'.png',crop_img)
           cv2.circle(frame1, (min_pix_dis[0],min_pix_dis[1]), 2, (0,0,255),4)
        elif min_pix_dis[0]<img_center[0] & min_pix_dis[1]>img_center[1] :
           crop_img = ndimage.rotate(crop_img, -180)
           mark=3 
           cv2.imwrite('cropped_image_ID-'+str(i)+'.png',crop_img)
           img_st,img_end,img_circle,poses_3d=humanSkeleton(crop_img,img_st,img_end,img_circle,poses_3d,frame1,x_dr,y_dr,min_pix_dis)
           print('3')  
           #cv2.imwrite('cropped_image_ID-'+str(i)+'.png',crop_img)
           frame1,frame_top=drawSkeleton2(frame1,x_dr,y_dr,img_st,img_end,img_circle,frame_top)           
           cv2.circle(frame1, (min_pix_dis[0],min_pix_dis[1]), 2, (0,0,255),4)
        elif min_pix_dis[0]<img_center[0] & min_pix_dis[1]<img_center[1] :
           crop_img = ndimage.rotate(crop_img, -90)
           mark=4
           cv2.imwrite('cropped_image_ID-'+str(i)+'.png',crop_img) 
           img_st,img_end,img_circle,poses_3d=humanSkeleton(crop_img,img_st,img_end,img_circle,poses_3d,frame1,x_dr,y_dr,min_pix_dis)
           frame1,frame_top=drawSkeleton4(frame1,x_dr,y_dr,img_st,img_end,img_circle,frame_top)
           cv2.circle(frame1, (min_pix_dis[0],min_pix_dis[1]), 2, (0,0,255),4)
           print('4')           
        #print(min_pix_dis)
        #min_pix_dis[0]=int(min_pix_dis[0])
        #min_pix_dis[1]=int(min_pix_dis[1])
        pixel = groundProjectPoint(min_pix_dis)
        #im_crop = frame1.crop((x, x+w, y, y+h))
        #im_crop.save('im_pillow_crop.jpg', quality=95)
        #print('Moving Object ID-{}\n'.format(i),'World Coordinate [X,Y,Z] in cm\n',pixel)
        
        #cv2.putText(frame1,str(pixel[1]) , (100, 300), cv2.FONT_HERSHEY_SIMPLEX,
                   # 1, (0, 0, 255), 3)
        cv2.circle(frame1, (min_pix_dis[0],min_pix_dis[1]), 2, (0,0,255),4)
        #cv2.putText(frame1, "Coordinate: {}".format('Movement'), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    #1, (0, 0, 255), 3)
        #cv2.putText(frame1, "ID: {}".format(str(i)), (c_x,c_y), cv2.FONT_HERSHEY_SIMPLEX,
                    #1, (0, 0, 255), 3)
        #cv2.putText(frame1, "Status: {}".format('Movement Found-'+str(i)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
        #            1, (0, 0, 255), 3)
        i=i+1 
    #cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)
    #print("Axes:X-",x,"Axes:Y-",y) 
    #print("Center:X-",cx,"Center:Y-",cy) 
    #image = cv2.resize(frame1, (1280,720))
    #out.write(image)
    #frame1 = cv2.line(frame1, (0,img_center[1]), (img_width,img_center[1]), (0,0,255), 3)
    #frame1 = cv2.line(frame1, (img_center[0],0), (img_center[0],img_height), (0,0,255), 3) 
    #cv2.imwrite("new_feed.png", frame1)
    #frame1 = frame2
    #ret, frame2 = cap.read()
    #frame2 = cv2.remap(frame2, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    #if cv2.waitKey(1) & 0xFF == ord(' '):
    #    break
    time=float(time)
    time2=float(time2)
    time+=0.1

    time=round(time,2)

cv2.destroyAllWindows()
#cap.release()
#out.release()

