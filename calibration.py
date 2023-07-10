import numpy as np 
import cv2
from tqdm import tqdm

# Set the path to the images captured by the left and right cameras
pathL = "./data/checkboard_10x7/stereoL/"
pathR = "./data/checkboard_10x7/stereoR/"

print("Extracting image coordinates of respective 3D pattern ....\n")

# Termination criteria for refining the detected corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


objp = np.zeros((10*7,3), np.float32)
objp[:,:2] = np.mgrid[0:10,0:7].T.reshape(-1,2) #* 18.9
img_ptsL = []
img_ptsR = []
obj_pts = []

for i in tqdm(range(0,75)):
	imgL = cv2.imread(pathL+f"left_{i}.png")
	imgR = cv2.imread(pathR+f"right_{i}.png")
	imgL_gray = cv2.imread(pathL+f"left_{i}.png",0)
	imgR_gray = cv2.imread(pathR+f"right_{i}.png",0)

	outputL = imgL.copy()
	outputR = imgR.copy()

	retR, cornersR =  cv2.findChessboardCorners(outputR,(10,7),None)
	retL, cornersL = cv2.findChessboardCorners(outputL,(10,7),None)

	if retR and retL:
		obj_pts.append(objp)
		cornersR2 = cv2.cornerSubPix(imgR_gray,cornersR,(11,11),(-1,-1),criteria)
		cornersL2 = cv2.cornerSubPix(imgL_gray,cornersL,(11,11),(-1,-1),criteria)
		cv2.drawChessboardCorners(outputR,(10,7),cornersR2,retR)
		cv2.drawChessboardCorners(outputL,(10,7),cornersL2,retL)
		cv2.imshow('cornersR',outputR)
		cv2.imshow('cornersL',outputL)
		cv2.waitKey(0)

		img_ptsL.append(cornersL2)
		img_ptsR.append(cornersR2)

# Calibrating left camera
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(obj_pts,img_ptsL,imgL_gray.shape[::-1],None,None)
hL,wL= imgL_gray.shape[:2]
new_mtxL, roiL= cv2.getOptimalNewCameraMatrix(mtxL,distL,(wL,hL),0,(wL,hL))
print('왼쪽 카메라 메트릭트', new_mtxL)

# Calibrating right camera
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(obj_pts,img_ptsR,imgR_gray.shape[::-1],None,None)
hR,wR= imgR_gray.shape[:2]
new_mtxR, roiR= cv2.getOptimalNewCameraMatrix(mtxR,distR,(wR,hR),0,(wR,hR))
print('오른쪽 카메라 메트릭트', new_mtxR)


flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC

# Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
# Hence intrinsic parameters are the same 
criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
retS, new_mtxL, distL, new_mtxR, distR, Rot, Trns, Emat, Fmat = cv2.stereoCalibrate(obj_pts,
                                                          img_ptsL,
                                                          img_ptsR,
                                                          new_mtxL,
                                                          distL,
                                                          new_mtxR,
                                                          distR,
                                                          imgL_gray.shape[::-1],
                                                          criteria_stereo,
                                                          flags)

print('스테레오 왼쪽 카메라 메트릭트', new_mtxL)
print('스테레오 오른쪽 카메라 메트릭트', new_mtxR)

# Once we know the transformation between the two cameras we can perform stereo rectification
# StereoRectify function
rectify_scale= 0.0 # if 0 image croped, if 1 image not croped
rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR= cv2.stereoRectify(new_mtxL, distL, new_mtxR, distR,
									  imgL_gray.shape[::-1], Rot, Trns, alpha=0)


# Use the rotation matrixes for stereo rectification and camera intrinsics for undistorting the image
# Compute the rectification map (mapping between the original image pixels and
# their transformed values after applying rectification and undistortion) for left and right camera frames
Left_Stereo_Map= cv2.initUndistortRectifyMap(new_mtxL, distL, rect_l, proj_mat_l,
                                             imgL_gray.shape[::-1], cv2.CV_16SC2)
Right_Stereo_Map= cv2.initUndistortRectifyMap(new_mtxR, distR, rect_r, proj_mat_r,
                                              imgR_gray.shape[::-1], cv2.CV_16SC2)
Left_Stereo_Map_x = Left_Stereo_Map[0]
Left_Stereo_Map_y = Left_Stereo_Map[1]
Right_Stereo_Map_x = Right_Stereo_Map[0]
Right_Stereo_Map_y = Right_Stereo_Map[1]

Left_nice= cv2.remap(imgL_gray,
					Left_Stereo_Map_x,
					Left_Stereo_Map_y,
					cv2.INTER_LANCZOS4,
					cv2.BORDER_CONSTANT,
					0)

# Applying stereo image rectification on the right image
Right_nice= cv2.remap(imgR_gray,
					Right_Stereo_Map_x,
					Right_Stereo_Map_y,
					cv2.INTER_LANCZOS4,
					cv2.BORDER_CONSTANT,
					0)

print("Saving paraeters ......")
cv_file = cv2.FileStorage("./data/stereo_rectify_maps.xml", cv2.FILE_STORAGE_WRITE)
cv_file.write("Left_Stereo_Map_x",Left_Stereo_Map[0])
cv_file.write("Left_Stereo_Map_y",Left_Stereo_Map[1])
cv_file.write("Right_Stereo_Map_x",Right_Stereo_Map[0])
cv_file.write("Right_Stereo_Map_y",Right_Stereo_Map[1])
cv_file.write("intrinsic_matrix_L", new_mtxL)
cv_file.write("intrinsic_matrix_R", new_mtxR)
cv_file.write("dist_L", distL)
cv_file.write("dist_R", distR)
cv_file.write("R", Rot)
cv_file.write("T", Trns)
cv_file.write("E", Emat)
cv_file.write("F", Fmat)
cv_file.write("Q", Q)
cv_file.release()