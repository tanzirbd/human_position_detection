#BG3
from matplotlib import pyplot as plt
import cv2
import numpy as np
from matplotlib import pyplot as plt

#Clicker on image
def click_event(event, x, y, flags, param):
	if event == cv2.EVENT_LBUTTONDOWN:
		#Print distance to pixel (x, y)
		print ("X-axis=",x,"Y-axis=",y ,"Z-axis=", float(309.6/displ[y,x]), "meter.")

print ('Loading images...')
imgL = cv2.imread('test/left_rectify_new8_chessboard.png',0)
imgR = cv2.imread('test/right_rectify_new8_chessboard.png',0)

#Disparity settings
window_size = 3
min_disp =0
num_disp = 128 -min_disp
matcher_left = cv2.StereoSGBM_create(
	blockSize = 5,
	numDisparities = num_disp,
	minDisparity = min_disp,
	P1 = 8*3*window_size**2,
	P2 = 32*3*window_size**2,
	disp12MaxDiff = 1,
	uniquenessRatio = 15,
	speckleWindowSize = 0,
	speckleRange = 5,
	preFilterCap = 63,
	mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
	)
matcher_right = cv2.ximgproc.createRightMatcher(matcher_left)

# Filter
lmbda = 80000 #org 80000
sigma = 1.2 #org 1.2400,330
#visual_multiplier = cv2.setMouseCallback("Disparity", click_event)

wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=matcher_left)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

#Disparity calculation
print ('Calculating disparity...')
displ = matcher_left.compute(imgL, imgR) .astype(np.float32)
displ = np.int16(displ)
dispr = matcher_right.compute(imgR, imgL) .astype(np.float32)
dispr = np.int16(dispr)

filteredImg = wls_filter.filter(displ, imgL, None, dispr)
filteredImg = cv2.normalize(
    src=filteredImg,
    dst=filteredImg,
    beta=1,
    alpha=255,
   	norm_type=cv2.NORM_MINMAX,
	dtype=cv2.CV_8U
    )
filteredImg = np.uint8(filteredImg)

#Show images
#input E F G H
(x,y)=(578,1116)    
#fig=plt.figure(figsize=(5,5))
np.savez('disparity_data.npz',displ=displ)
#q=plt.imshow(displ, cmap="plasma")
#plt.savefig('q.png')
print('Disparity\n',displ[y,x])
#right=[y,x+displ[y,x]]
#print(right)
cv2.imwrite('Disparity.png', filteredImg)
cv2.imwrite('dsada.png', dispr)
#cv2.imshow('Left', imgL)
#cv2.imshow('Right', imgR)
#cv2.imwrite('Both_Images.png', np.hstack([imgL, imgR]))
#cv2.setMouseCallback("Disparity", click_event)


cv2.waitKey(0)
cv2.destroyAllWindows
