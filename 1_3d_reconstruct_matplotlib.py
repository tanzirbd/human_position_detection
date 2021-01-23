#BG3

import cv2
import numpy as np
from matplotlib import pyplot as plt
import numpy as np

#from matplotlib.patches import Circle
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import mpl_toolkits.mplot3d.art3d as art3d
#Clicker on image
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111, projection='3d')#,aspect='equal')


print ('Loading images...')
imgL = cv2.imread('2_3d_reconstruction/input_image/left_rectify.png')
imgR = cv2.imread('2_3d_reconstruction/input_image/right_rectify.png')
img = cv2.imread('2_3d_reconstruction/input_image/right_rectify.png')
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

#Disparity settings
window_size = 3
min_disp = 1
num_disp = 16*2
stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        #SADWindowSize=window_size,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        disp12MaxDiff=1,
        P1=8*3*window_size**2,
        P2=32*3*window_size**2,
        #fullDP=False
    )
displ=stereo.compute(imgL, imgR).astype(np.float32) / 16.0
#print(displ[1116,446])
cv2.imshow('dsa',displ)

q = np.array([[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00, -7.29742737e+02],
 [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00, -5.92018181e+02],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  7.70295212e+02],
 [ 0.00000000e+00,  0.00000000e+00,  2.00621811e-01, -0.00000000e+00]])


points = cv2.reprojectImageTo3D(displ, q).reshape(-1, 3)
colors = img.reshape(-1, 3)
print(points.shape)
disp=displ.reshape(-1)
print(disp.min())
mask = (
        (disp > disp.min()) &
        np.all(~np.isnan(points), axis=1) &
        np.all(~np.isinf(points), axis=1)
    )
point=points[mask]
img=colors[mask]
print(point.shape)
for i in range(0,1000):

    X=point[i,0]/1000
    Y=point[i,1]/1000
    Z=point[i,2]/1000
    x = [X]
    y = [Y]
    z = [Z]
    ax.scatter(x,y,z, s=5,c='b')


    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    #plt.savefig("3Dpathpatch.png", dpi=100,transparent = False)

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows
