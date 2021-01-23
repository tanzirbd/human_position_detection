#%matplotlib notebook
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.patches import Circle
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import mpl_toolkits.mplot3d.art3d as art3d
import os
from os import path
import openpyxl
import xlrd 

SKELETON_EDGES = np.array([[11, 10], [10, 9], [9, 0], [0, 3], [3, 4], [4, 5], [0, 6], [6, 7], [7, 8], [0, 12],
                               [12, 13], [13, 14], [0, 1], [1, 15], [15, 16], [1, 17], [17, 18]])

#Give image starting and ending numbers 

time=12.1     #starting image time
for a in range(122,124):
    a=str(a)
    print(a)
    time=str(time) 
    path= '15_new_data_excel/new_data/'
    data1 = os.path.exists(path+'image'+a+'_data_ID-1_time-'+time+'.xls') 
    data2 = os.path.exists(path+'image'+a+'_data_ID-2_time-'+time+'.xls')
    data3 = os.path.exists(path+'image'+a+'_data_ID-3_time-'+time+'.xls')
    print(data1,data2,data3)
#image123_data_ID-1_time-12.2
    if (data1):
       loc = (path+'image'+a+'_data_ID-1_time-'+time+'.xls') 
       wb = xlrd.open_workbook(loc) 
       sheet = wb.sheet_by_index(0) 
       sheet.cell_value(4, 12) 
       sheet.cell_value(4, 13)
       sheet.cell_value(4, 14)
       x=[]
       y=[]
       z=[]
       for i in range(4,23):
           p=(sheet.cell_value(i, 12))
           q=(sheet.cell_value(i, 13))
           r=(sheet.cell_value(i, 14))
           p=float(p)
           q=float(q)
           r=float(r)
           x.append(p)
           y.append(q)
           z.append(r)
    if (data2):
       loc1 =(path+'image'+a+'_data_ID-2_time-'+time+'.xls')
       wb1 = xlrd.open_workbook(loc1)  
       sheet1 = wb1.sheet_by_index(0) 
       sheet1.cell_value(4, 12) 
       sheet1.cell_value(4, 13)
       sheet1.cell_value(4, 14)
       x1=[]
       y1=[]
       z1=[]
       for i in range(4,23):
           p1=(sheet1.cell_value(i, 12))
           q1=(sheet1.cell_value(i, 13))
           r1=(sheet1.cell_value(i, 14))
           p1=float(p1)
           q1=float(q1)
           r1=float(r1)
           x1.append(p1)
           y1.append(q1)
           z1.append(r1)  
    elif (data3):
       loc2 = (path+'image'+a+'_data_ID-3_time-'+time+'.xls')
       wb2 = xlrd.open_workbook(loc2) 
       sheet2 = wb2.sheet_by_index(0) 
       sheet2.cell_value(4, 12) 
       sheet2.cell_value(4, 13)
       sheet2.cell_value(4, 14)
       x2=[]
       y2=[]
       z2=[]
       for i in range(4,23):
           p2=(sheet2.cell_value(i, 12))
           q2=(sheet2.cell_value(i, 13))
           r2=(sheet2.cell_value(i, 14))
           p2=float(p2)
           q2=float(q2)
           r2=float(r2)
           x2.append(p2)
           y2.append(q2)
           z2.append(r2) 
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection='3d')#,aspect='equal')    


    for i in range(0,17):
         m=SKELETON_EDGES[i,0]
         n=SKELETON_EDGES[i,1]
         if(data1):
            line= art3d.Line3D([x[m],x[n]],[y[m],y[n]],[z[m],z[n]], color='g') 
            ax.add_line(line)
         if(data2):
            line2= art3d.Line3D([x1[m],x1[n]],[y1[m],y1[n]],[z1[m],z1[n]], color='g') 
            ax.add_line(line2)
         if(data3):
            line3= art3d.Line3D([x2[m],x2[n]],[y2[m],y2[n]],[z2[m],z2[n]], color='g') 
            ax.add_line(line3)

    ax.set_xlim(-2, 3)
    ax.set_ylim(-2, 3)
    ax.set_zlim(0, 2)


    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    path2='17_3d_image_new/'
    plt.savefig(path2+'image_'+a+'_3d_image_time-'+time+'.png')

    plt.show()

    time=float(time)
    time+=0.1
    time=round(time,2)
#end
