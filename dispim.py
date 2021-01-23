#import cv2
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
#import xlwt 
#from xlwt import Workbook 
#from PIL import Image

#from matplotlib.patches import Circle
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import mpl_toolkits.mplot3d.art3d as art3d


print ('Loading images...')
# 画像の読み込み
#img = np.array( Image.open('./disparity_data.npz') )

disp_ar=np.load('./disparity_data.npz')
#world_ar=np.load('./world_coordinate_data.npz')

print(disp_ar.files)
#print(world_ar.files)
#p3d=world_ar['points_3d']
#print('p3d shape=',p3d.shape)

disp1=disp_ar['displ']
#print(disp1)

idy,idx=disp1.shape
print('idx=',idx,'idy=',idy)

#print(disp1[1,1])
disp2=disp1.tolist
print(disp2)

ipo=idx*idy
print(ipo)
kp=0

x=[]
y=[]
z=[]

i1=1000
i2=1300
j1=500
j2=1000

#for i in range(0,idx,30):
for i in range(i1,i2,5):
   #print(i, disp2[i,1])
   #for j in range(0,idy,30): 
   for j in range(j1,j2,5):
      #print(i,j,kp,disp1[j,i])    
      x.append(i)
      y.append(j)
      z.append(disp1[j,i])      
      #print(kp,x[kp],y[kp],z[kp])
      kp=kp+1
#print(disp_ar['displ'][0,:])
#print(world_ar['points_3d'])

#Clicker on image
fig = plt.figure(figsize=(idx,idy))
ax = fig.add_subplot(111, projection='3d')#,aspect='equal')
ax.scatter(x,y,z, s=5,c='b')

#ax.set_xlim(0, idx)
ax.set_xlim(i1, i2)
#ax.set_ylim(0, idy)
ax.set_ylim(j1, j2)
ax.set_zlim(-16, 2300)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

plt.show()



