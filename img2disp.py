import numpy as np 
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
import sys

# np.set_printoptions(threshold=sys.maxsize)

def print_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # 마우스 왼쪽 버튼을 눌렀을 때
        print(f'클릭한 픽셀의 좌표는 ({x}, {y}) 입니다.')

# 카메라 파라미터
focal_length_L = 1427.3084
baseline = 60.0
epsilon = 1e-6

# Reading the mapping values for stereo image rectification
cv_file = cv2.FileStorage("./data/stereo_rectify_maps.xml", cv2.FILE_STORAGE_READ)
Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
camera_matrix_L = cv_file.getNode("intrinsic_matrix_L").mat()
camera_matrix_R = cv_file.getNode("intrinsic_matrix_R").mat()
Q = cv_file.getNode("Q").mat()
cv_file.release()

# print('camera matrix L', camera_matrix_L)
# print('camera matrix R', camera_matrix_R)
# print('original Q matrix', Q)

# Q matrix = [[1,0,0,-Cx],[0,1,0,-Cy], [0,0,0,f], [0,0,-1/Tx, (Cx-Cx')/Tx]]
# Cx: 왼쪽 카메라 x 주점
# Cy: 왼쪽 카메라 y 주점
# Cx': 오른쪽 카메라 x 주점
# Tx: 카메라 렌즈 기준선(baseline)


# Q = np.float32([[1, 0, 0, -968.8848659],
#                [0, -1, 0, -545.83119],
#                [0, 0, 1428.08808, 0],
#                [0, 0, 1/60, 1]])

Q = np.float32([[1, 0, 0, -968.8848659],
               [0, 1, 0, -545.83119],
               [0, 0, 0, 1428.08808],
               [0, 0, 1/60, (968.8848659-938.823154)/60]])

# Q = np.float32([[1/(5.07e-6), 0, 0, -968.8848659],
#                [0, 1/(3.38e-6), 0, -545.83119],
#                [0, 0, 0, 1428.08808],
#                [0, 0, 1/60, (968.8848659-938.823154)/60]])

# print('edited Q matrix', Q)

# These parameters can vary according to the setup
# Keeping the target object at max_dist we store disparity values
# after every sample_delta distance.
max_dist = 1000 # max distance to keep the target object (in cm)
min_dist = 50 # Minimum distance the stereo setup can measure (in cm)
sample_delta = 40 # Distance between two sampling points (in cm)

Z = max_dist

# Reading the stored the StereoBM parameters
cv_file = cv2.FileStorage("./data/tuned_depth_parameter.xml", cv2.FILE_STORAGE_READ)
minDisparity = int(cv_file.getNode("minDisparity").real())
numDisparities = int(cv_file.getNode("numDisparities").real())
blockSize = int(cv_file.getNode("blockSize").real())
P1 = int(cv_file.getNode("P1").real())
P2 = int(cv_file.getNode("P2").real())
disp12MaxDiff = int(cv_file.getNode("disp12MaxDiff").real())
preFilterCap = int(cv_file.getNode("preFilterCap").real())
uniquenessRatio = int(cv_file.getNode("uniquenessRatio").real())
speckleWindowSize = int(cv_file.getNode("speckleWindowSize").real())
speckleRange = int(cv_file.getNode("speckleRange").real())
Mode = int(cv_file.getNode("mode").real())
M = cv_file.getNode("M").real()
cv_file.release()
	

# Set window size
cv2.namedWindow('rectified_left_image',cv2.WINDOW_NORMAL)
cv2.resizeWindow('rectified_left_image',400,400)
cv2.namedWindow('rectified_right_image',cv2.WINDOW_NORMAL)
cv2.resizeWindow('rectified_right_image',400,400)
cv2.namedWindow('disp',cv2.WINDOW_NORMAL)
cv2.resizeWindow('disp',400,400)
cv2.setMouseCallback('disp', print_coordinates)
cv2.namedWindow('wls_filtered_displ',cv2.WINDOW_NORMAL)
cv2.resizeWindow('wls_filtered_displ',400,400)
# cv2.namedWindow('depth_map_left',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('depth_map_left',400,400)

# wls filter parameter
lmbda = 8000
sigma = 1.5

# Creating an object of StereoSGBM algorithm
left_matcher = cv2.StereoSGBM_create()
right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

# generate wls filter instance
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left = left_matcher)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

# Capturing and storing left and right camera images
imgL = cv2.imread('C:/Users/MSDL-DESK-02/Desktop/pyStereo/data/checkboard_10x7/stereoL/left_0.png')
imgR = cv2.imread('C:/Users/MSDL-DESK-02/Desktop/pyStereo/data/checkboard_10x7/stereoR/right_0.png')

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

# Setting the updated parameters before computing disparity map
left_matcher.setMinDisparity(minDisparity)
left_matcher.setNumDisparities(numDisparities)
left_matcher.setBlockSize(blockSize)
left_matcher.setP1(P1)
left_matcher.setP2(P2)
left_matcher.setDisp12MaxDiff(disp12MaxDiff)
left_matcher.setPreFilterCap(preFilterCap)
left_matcher.setUniquenessRatio(uniquenessRatio)
left_matcher.setSpeckleWindowSize(speckleWindowSize)
left_matcher.setSpeckleRange(speckleRange)
left_matcher.setMode(0)

# Calculating disparity using the StereoSGBM algorithm
displ = left_matcher.compute(Left_nice, Right_nice).astype(np.float32)
dispr = right_matcher.compute(Right_nice, Left_nice).astype(np.float32)

# Scaling down the disparity values and normalizing them
displ = (displ/16.0 - minDisparity)/numDisparities
displ += epsilon
dispr = (dispr/16.0 - minDisparity)/numDisparities
dispr += epsilon
filtered_displ = wls_filter.filter(displ, Left_nice, None, dispr)

# Convert disparity map to point cloud
points = cv2.reprojectImageTo3D(filtered_displ, Q)
mask = displ > displ.min()
colors = cv2.cvtColor(Left_nice, cv2.COLOR_BGR2RGB)
out_points = points[mask]
out_colors = colors[mask]
norm_out_colors = out_colors/255.0
print("out_points",out_points)

# Create Open3D point cloud
point_cloud1 = o3d.geometry.PointCloud()
point_cloud1.points = o3d.utility.Vector3dVector(out_points)
point_cloud1.colors = o3d.utility.Vector3dVector(norm_out_colors)
point_cloud2 = o3d.geometry.PointCloud()
point_cloud2.points = o3d.utility.Vector3dVector(out_points)

# Create coordinate frame at 0, 0, 0
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=500, origin=[0, 0, 0])

# Create a list of geometries
geometries1 = [point_cloud1, coordinate_frame]
geometries2 = [point_cloud2, coordinate_frame]

# Visualize the point cloud and coordinate frame
o3d.visualization.draw_geometries(geometries1)
o3d.visualization.draw_geometries(geometries2)
o3d.io.write_point_cloud('./result/point_cloud.ply', point_cloud1)

# Displaying the disparity map
cv2.imshow("rectified_left_image",Left_nice)
cv2.imshow("rectified_right_image",Right_nice)
cv2.imshow("disp",displ)
cv2.imshow('wls_filtered_displ', filtered_displ)

cv2.waitKey(0)
cv2.destroyAllWindows()

# cv2.imwrite('./result/Left_nice.jpg', Left_nice)
# cv2.imwrite('./result/Right_nice.jpg', Right_nice)
# cv2.imwrite('./result/Disparity_left.jpg', displ)