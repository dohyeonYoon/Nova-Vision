import numpy as np 
import cv2
 
# Check for left and right camera IDs
# These values can change depending on the system

# Open the left and right cameras
CamL= cv2.VideoCapture(1)
CamR= cv2.VideoCapture(0)

# Set the resolution for each camera
CamL.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
CamL.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
CamR.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
CamR.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if CamL.isOpened():
  print('width: {}, height : {}'.format(CamL.get(3), CamL.get(4)))

if CamR.isOpened():
  print('width: {}, height : {}'.format(CamR.get(3), CamR.get(4)))
 
# Reading the mapping values for stereo image rectification
cv_file = cv2.FileStorage("./data/stereo_rectify_maps.xml", cv2.FILE_STORAGE_READ)
Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
cv_file.release()
 
def nothing(x):
    pass
 
cv2.namedWindow('disp',cv2.WINDOW_NORMAL)
cv2.resizeWindow('disp',1280,720)

cv2.createTrackbar('min_disparity','disp',0,20,nothing)
cv2.createTrackbar('num_disparity','disp',1,20,nothing)
cv2.createTrackbar('blockSize','disp',5,50,nothing)
cv2.createTrackbar('P1','disp',0,1,nothing)
cv2.createTrackbar('P2','disp',0,1,nothing)
cv2.createTrackbar('disp12maxdiff','disp',0,25,nothing)
cv2.createTrackbar('preFilterCap','disp',0,20,nothing)
cv2.createTrackbar('uniquenessRatio','disp',0,15,nothing)
cv2.createTrackbar('speckleWindowSize','disp',0,20,nothing)
cv2.createTrackbar('speckleRange','disp',0,5,nothing)
cv2.createTrackbar('mode','disp',0,3,nothing)
# Creating an object of StereoBM algorithm
stereo = cv2.StereoSGBM_create()
 
while True:
 
  # Capturing and storing left and right camera images
  retL, imgL= CamL.read()
  retR, imgR= CamR.read()
   
  # Proceed only if the frames have been captured
  if retL and retR:
    imgR_gray = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)
    imgL_gray = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
 
    # Applying stereo image rectification on the left image
    Left_nice= cv2.remap(imgL,
              Left_Stereo_Map_x,
              Left_Stereo_Map_y,
              cv2.INTER_LANCZOS4,
              cv2.BORDER_CONSTANT,
              0)
     
    # Applying stereo image rectification on the right image
    Right_nice= cv2.remap(imgR,
              Right_Stereo_Map_x,
              Right_Stereo_Map_y,
              cv2.INTER_LANCZOS4,
              cv2.BORDER_CONSTANT,
              0)

    # Updating the parameters based on the trackbar positions
    min_disparity = cv2.getTrackbarPos('min_disparity','disp')
    num_disparity = cv2.getTrackbarPos('num_disparity','disp')*16
    blockSize = cv2.getTrackbarPos('blockSize','disp')*2 + 3
    P1 = cv2.getTrackbarPos('P1','disp')*8*3*blockSize **2
    P2 = cv2.getTrackbarPos('P2','disp')*32*3*blockSize **2
    disp12maxdiff = cv2.getTrackbarPos('disp12maxdiff','disp')
    prefiltercap = cv2.getTrackbarPos('preFilterCap','disp')
    uniqueness_ratio = cv2.getTrackbarPos('uniquenessRatio','disp')
    speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize','disp')*10
    speckleRange = cv2.getTrackbarPos('speckleRange','disp')
    mode = cv2.getTrackbarPos('mode','disp') # 0:SGBM, 1:HH, 2:SGBM_3WAY, 3:HH4

     
    # Setting the updated parameters before computing disparity map
    stereo.setMinDisparity(min_disparity)
    stereo.setNumDisparities(num_disparity)
    stereo.setBlockSize(blockSize)
    stereo.setP1(P1)
    stereo.setP2(P2)
    stereo.setDisp12MaxDiff(disp12maxdiff)
    stereo.setPreFilterCap(prefiltercap)
    stereo.setUniquenessRatio(uniqueness_ratio)
    stereo.setSpeckleWindowSize(speckleWindowSize)
    stereo.setSpeckleRange(speckleRange)
    stereo.setMode(mode)
    
    # Calculating disparity using the StereoBM algorithm
    disparity = stereo.compute(Left_nice,Right_nice)
 
    # Converting to float32 
    disparity = disparity.astype(np.float32)
 
    # Scaling down the disparity values and normalizing them 
    disparity = (disparity/16.0 - min_disparity)/num_disparity
 
    # Displaying the disparity map
    cv2.imshow("disp",disparity)

    print(disparity.shape)
 
    # Close window using esc key
    if cv2.waitKey(1) == 27:
      break
   
  else:
    CamL= cv2.VideoCapture(1)
    CamR= cv2.VideoCapture(0)
  
print("Saving depth estimation paraeters ......")

cv_file = cv2.FileStorage("./data/test.xml", cv2.FILE_STORAGE_WRITE)
cv_file.write("minDisparity",min_disparity)
cv_file.write("numDisparities",num_disparity)
cv_file.write("blockSize",blockSize)
cv_file.write("P1", P1)
cv_file.write("P2", P2)
cv_file.write("disp12MaxDiff",disp12maxdiff)
cv_file.write("preFilterCap",prefiltercap)
cv_file.write("uniquenessRatio",uniqueness_ratio)
cv_file.write("speckleWindowSize",speckleWindowSize)
cv_file.write("speckleRange",speckleRange)
cv_file.write("mode", mode)
cv_file.write("M",39.075)
cv_file.release()