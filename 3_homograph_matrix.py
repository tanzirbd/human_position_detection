import cv2

import numpy as np

if __name__ == '__main__' :

    im_src = cv2.imread('10_homograph_matrix_input_output_file/input_image/left_rectify.png')

    # Four corners of the book in source image

    pts_src = np.array([[432, 997], [462, 494], [1080, 523],[1058, 1010]])

 

 

    # Read destination image.

    im_dst = cv2.imread('10_homograph_matrix_input_output_file/input_image/right_rectify.png')

    # Four corners of the book in destination image.

    pts_dst = np.array([[340, 992],[374, 490],[980, 523],[958, 1012]])

 

    # Calculate Homography

    h, status = cv2.findHomography(pts_src, pts_dst)
    
    a=np.array([[954,900]],dtype='float32')
    a=np.array([a])
    #print(a.shape)
    pointsOut=cv2.perspectiveTransform(a,h)
    print('perspective',pointsOut)

    im_out = cv2.warpPerspective(im_dst, h, (im_src.shape[1],im_dst.shape[0])) 
    print(h)
    # Warp source image to destination based on homography
    for i in range(len(pts_src)):
     pt1 = np.array([pts_src[i][0], pts_src[i][1], 1])
     pt1 = pt1.reshape(3, 1)
     #pts_dst=(pts_src[0,0],pts_src[0,1],1)
     pt2 = np.dot(h, pt1)
     #print(pt2)
    # Display images

     cv2.imwrite("10_homograph_matrix_input_output_file/output_image/Warped_source_Image_right.png", im_out)

 

    cv2.waitKey(0)
