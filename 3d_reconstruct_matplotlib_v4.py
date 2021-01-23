#BG3

import cv2
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import xlwt 
from xlwt import Workbook 

#from matplotlib.patches import Circle
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import mpl_toolkits.mplot3d.art3d as art3d
#Clicker on image
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111, projection='3d')#,aspect='equal')

def click_event(event, x, y, flags, param):
	if event == cv2.EVENT_LBUTTONDOWN:
		#Print distance to pixel (x, y)
		print ("X-axis=",x,"Y-axis=",y ,"Z-axis=", float(309.6/displ[y,x]), "meter.")

print ('Loading images...')
imgL = cv2.imread('11_disparity/input/left_rectify.png',0)
imgR = cv2.imread('11_disparity/input/right_rectify.png',0)
img = cv2.imread('11_disparity/input/right_rectify.png')
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

#Disparity settings
window_size = 3
min_disp =0
num_disp = 128
matcher_left = cv2.StereoSGBM_create(
	blockSize = 3,
	numDisparities = num_disp,
	minDisparity = min_disp,
	P1 = 8*3*window_size**2,
	P2 = 32*3*window_size**2,
	disp12MaxDiff = 1,
	uniquenessRatio = 15,
	speckleWindowSize = 50,
	speckleRange = 5,
	preFilterCap = 63,
	mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
	)
matcher_right = cv2.ximgproc.createRightMatcher(matcher_left)
displ = matcher_left.compute(imgL, imgR)#.astype(np.float32)/16.0
displ = np.int16(displ)
dispr = matcher_right.compute(imgR, imgL)#.astype(np.float32)/16.0
#dispr = np.int16(dispr)
(x,y)=(462, 494)
print(displ[y,x])
np.savez('disparity_data.npz',displ=displ)
files=np.load('disparity_data.npz')
#displ1=files['displ']
#print('disparity',displ1.shape)
cv2.imshow('dsa',displ)

q = np.array([[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00, -7.29742737e+02],
 [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00, -5.92018181e+02],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  7.70295212e+02],
 [ 0.00000000e+00,  0.00000000e+00,  2.00621811e-01, -0.00000000e+00]])


points_3d = cv2.reprojectImageTo3D(displ, q)/100
points=points_3d
print(points.shape)

#sheet1 = wb.add_sheet('sheet_1')
#sheet1.write(0, 0, 'World Coordinate')
#sheet1.write(1, 2, 'measure the value in m')
#sheet1.write(2, 2, 'Row 04-1204 is image Height and Row 1205-2805 is image Wiedth ')
(x1,y1)=(1104, 1096)
print(points[y1,x1])
#points=points.reshape(-1, 3)
#points=points/100
colors = img.reshape(-1, 3)
np.savez('world_coordinate_data.npz',points_3d=points_3d)
files=np.load('world_coordinate_data.npz')
points_3d1=files['points_3d']
print('dsdsada',points_3d1.shape)

#wb = Workbook() 
#sheet1 = wb.add_sheet('sheet_1')
#sheet1.write(0, 0, 'World Coordinate')
#sheet1.write(1, 2, 'measure the value in m')
#sheet1.write(2, 2, 'Here row 04-1204 is image Height and 1205-2805 is image width.')
#for m in range(0,900):
#    X=float(points[m,0]/1000)
#    Y=float(points[m,1]/1000)
#    Z=float(points[m,2]/1000)
#    sheet1.write((m+4), 2, X)
#    sheet1.write((m+4), 3, Y)
#    sheet1.write((m+4), 4, Z)
#wb.save('world_data.xls')

#print(points.shape)

#Removeing holes
disp=displ.reshape(-1)
#print(disp.min())
mask = (
        (disp > disp.min()) &
        np.all(~np.isnan(points), axis=1) &
        np.all(~np.isinf(points), axis=1)
    )
point=points[mask]
img=colors[mask]
#print(point.shape)
#X1=[]
#Y1=[]
#Z1=[]

#for i in range (0,1920000):
#    X1=points_3d1[i,0]
#    Y1=points_3d1[i,1]
#    Z1=points_3d1[i,2]
#    X1.append(X1)
#    Y1.append(Y1)
#    Z1.append(Z1)

X=points_3d1[:, 0::4]
Y=points_3d1[:, 1::4]
Z=points_3d1[:, 2::4]

ax.scatter(X,Y,Z, s=5,c='b')


ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(-5, 5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.savefig('3d_reconstruct.png')

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows
