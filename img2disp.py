import numpy as np 
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
from natsort import natsorted
import os

def print_coordinates(event, x, y, flags, param):
    ''' opencv window에 마우스 클릭시 해당 마우스 포인터의 픽셀좌표 출력하는 함수
   
    Args: 
        event: 마우스 클릭 이벤트
        x: 마우스의 x 좌표
        y: 마우스의 y 좌표

    Return:
        -
    '''
    if event == cv2.EVENT_LBUTTONDOWN:  # 마우스 왼쪽 버튼을 눌렀을 때
        print(f'클릭한 픽셀의 좌표는 ({x}, {y}) 입니다.')
    
    return


def write_pgm(image, file_name, points):
    ''' 3D point cloud를 입력받아 pgm file을 생성하는 함수
    Args:
        image: rectified 입력 이미지
        file_name: 저장될 pgm file 이름
        points: opencv reprojectimageto3d 함수의 결과로 나온 3D point cloud

    Return:
        - 
    '''
    # Set pgm file name. 
    pgm_name = file_name + '.pgm'

    # Return input image size.
    height, width, channel = image.shape

    # Extract Z value from 3D point cloud.
    points = points[:, :, 2].astype(int)

    # Convert numpy array to string with aligned digits
    points = '\n'.join([' '.join([str(num).rjust(5) for num in row]) for row in points])

    # Write pgm file
    with open(pgm_name, 'w', newline='') as f:
        f.write('P2\n')
        f.write(f'{width} {height}\n')
        f.write('65535\n')
        f.write(points)
    
    return


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

# Q matrix = [[1,0,0,-Cx],[0,1,0,-Cy], [0,0,0,f], [0,0,-1/Tx, (Cx-Cx')/Tx]]
# Cx: 왼쪽 카메라 x 주점
# Cy: 왼쪽 카메라 y 주점
# Cx': 오른쪽 카메라 x 주점
# Tx: 카메라 렌즈 기준선(baseline)

# Edit Q matrix
Q = np.float32([[1, 0, 0, -968.8848659],
               [0, -1, 0, -545.83119],
               [0, 0, 1428.08808, 0],
               [0, 0, 1/60, 1]])

# Q = np.float32([[1, 0, 0, -968.8848659],
#                [0, 1, 0, -545.83119],
#                [0, 0, 0, 1428.08808],
#                [0, 0, 1/60, (968.8848659-938.823154)/60]])

# Q = np.float32([[1/(5.07e-6), 0, 0, -968.8848659],
#                [0, 1/(3.38e-6), 0, -545.83119],
#                [0, 0, 0, 1428.08808],
#                [0, 0, 1/60, (968.8848659-938.823154)/60]])

# Reading the stored the StereoSGBM parameters
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
# cv2.namedWindow('rectified_left_image',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('rectified_left_image',400,400)
# cv2.namedWindow('rectified_right_image',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('rectified_right_image',400,400)
# cv2.namedWindow('disp',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('disp',400,400)
# cv2.setMouseCallback('disp', print_coordinates)
# cv2.namedWindow('wls_filtered_displ',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('wls_filtered_displ',400,400)

# wls filter parameter
lmbda = 8000
sigma = 1.5

# Generate StereoSGBM instance
left_matcher = cv2.StereoSGBM_create()
right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

# Generate wls filter instance
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left = left_matcher)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

# Read left and right camera images
imgL = cv2.imread('C:/Users/MSDL-DESK-02/Desktop/pyStereo/data/checkboard_10x7/stereoL/left_0.png')
imgR = cv2.imread('C:/Users/MSDL-DESK-02/Desktop/pyStereo/data/checkboard_10x7/stereoR/right_0.png')

# Applying stereo image rectification on the left,right image
Left_nice= cv2.remap(imgL,
                    Left_Stereo_Map_x,
                    Left_Stereo_Map_y,
                    cv2.INTER_LANCZOS4,
                    cv2.BORDER_CONSTANT,
                    0)

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
epsilon = 1e-6
displ = (displ/16.0 - minDisparity)/numDisparities
displ += epsilon
dispr = (dispr/16.0 - minDisparity)/numDisparities
dispr += epsilon
filtered_displ = wls_filter.filter(displ, Left_nice, None, dispr)

# Convert disparity map to point cloud
points = cv2.reprojectImageTo3D(filtered_displ, Q)
points = points.clip(0, np.inf)
mask = displ > displ.min()
colors = cv2.cvtColor(Left_nice, cv2.COLOR_BGR2RGB)
out_points = points[mask]
out_colors = colors[mask]
norm_out_colors = out_colors/255.0

# Create open3D point cloud instance
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

# Visualize point cloud and coordinate frame
o3d.visualization.draw_geometries(geometries1)
o3d.visualization.draw_geometries(geometries2)

# Displaying result
# cv2.imshow("rectified_left_image",Left_nice)
# cv2.imshow("rectified_right_image",Right_nice)
# cv2.imshow("disp",displ)
# cv2.imshow('wls_filtered_displ', filtered_displ)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# save pgm file
file_name = 'depth'
write_pgm(imgL, file_name, points)