import numpy as np 
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
from natsort import natsorted
import os
import time

# Read left and right camera images
input_path = './input'
output_path = './output'
input_file_list = natsorted(os.listdir(input_path))
filename_list = natsorted(list(set(os.path.splitext(i)[0] for i in input_file_list)))
file_name = filename_list[0][:-2]
imgL_name = input_path + '/' + filename_list[0] + '.png'
imgR_name = input_path + '/' + filename_list[1] + '.png'
imgL = cv2.imread(imgL_name)
imgR = cv2.imread(imgR_name)

# Reading parameters(image rectification, StereoSGBM)
cv_file = cv2.FileStorage("../data/stereo_rectify_maps.xml", cv2.FILE_STORAGE_READ)
parameter = cv2.FileStorage("../data/tuned_depth_parameter.xml", cv2.FILE_STORAGE_READ)

# camera parameter
fx_L = 1428.08808
fy_L = 1433.08303
cx_L = 968.8848659
cy_L = 545.83119
fx_R = 1416.79417
fy_R = 1423.20713
cx_R =  938.823154
cy_R = 532.871976
baseline = 60.0

def main():
    Left_nice, Right_nice = rectify_img(imgL, imgR, cv_file)
    displ, dispr, wls_filtered_displ, conf_map, bilateral_filtered_displ, numDisparities = compute_disparity(Left_nice, Right_nice, parameter)
    generate_depthmap(Left_nice, displ, wls_filtered_displ, bilateral_filtered_displ, numDisparities)
    wls_points, no_filtered_points, wls_filtered_points,bilateral_filtered_displ, geometries1, geometries2, geometries3 = generate_pcd(Left_nice, displ, wls_filtered_displ, bilateral_filtered_displ)
    visualize(Left_nice, Right_nice, displ, wls_filtered_displ, conf_map,bilateral_filtered_displ, geometries1, geometries2, geometries3)
    save_pgm(Left_nice, file_name, wls_points, output_path)
    save_img(Left_nice, file_name, output_path)
    return

def rectify_img(imgL, imgR, cv_file):
    '''원본이미지를 입력받아 이미지 왜곡보정하는 함수'''

    # Reading the mapping values for stereo image rectification
    Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
    Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
    Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
    Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
    camera_matrix_L = cv_file.getNode("intrinsic_matrix_L").mat()
    camera_matrix_R = cv_file.getNode("intrinsic_matrix_R").mat()
    Q = cv_file.getNode("Q").mat()
    cv_file.release()

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

    return Left_nice, Right_nice

def crop_img(array, numDisparities):
    '''왼쪽으로부터 numDisparities개의 픽셀은 배제하는 함수'''
    cropped_array = array[:,numDisparities:]

    return cropped_array

def compute_disparity(Left_nice, Right_nice, parameter):
    '''disparity map을 계산하는 함수'''

    # Load StereoSGBM parameter
    minDisparity = int(parameter.getNode("minDisparity").real())
    numDisparities = int(parameter.getNode("numDisparities").real())
    blockSize = int(parameter.getNode("blockSize").real())
    P1 = int(parameter.getNode("P1").real())
    P2 = int(parameter.getNode("P2").real())
    disp12MaxDiff = int(parameter.getNode("disp12MaxDiff").real())
    preFilterCap = int(parameter.getNode("preFilterCap").real())
    uniquenessRatio = int(parameter.getNode("uniquenessRatio").real())
    speckleWindowSize = int(parameter.getNode("speckleWindowSize").real())
    speckleRange = int(parameter.getNode("speckleRange").real())
    Mode = int(parameter.getNode("mode").real())
    M = parameter.getNode("M").real()
    parameter.release()

    # Generate StereoSGBM instance
    left_matcher = cv2.StereoSGBM_create()
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    
    # Set updated parameters before computing disparity map
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
    # displ = left_matcher.compute(Left_nice, Right_nice).astype(np.float32)
    # dispr = right_matcher.compute(Right_nice, Left_nice).astype(np.float32)
    epsilon = 1e-6
    displ = left_matcher.compute(Left_nice, Right_nice).astype(np.float32)
    dispr = right_matcher.compute(Right_nice, Left_nice).astype(np.float32)

    # displ = displ / 16.0
    # dispr = dispr / 16.0

    # displ +=epsilon
    # dispr +=epsilon

    print('displ은?', displ)

    # Scaling down the disparity values and normalizing them
    
    displ = (displ/16.0 - minDisparity)/numDisparities
    displ += epsilon
    dispr = (dispr/16.0 - minDisparity)/numDisparities
    dispr += epsilon

    # Apply wls filter to normalized displ
    lmbda = 8000
    sigma = 1.5
    # wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left = left_matcher)
    wls_filter = cv2.ximgproc.createDisparityWLSFilterGeneric(use_confidence=True)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    wls_filtered_displ = wls_filter.filter(displ, Left_nice, None, dispr)
    wls_filtered_displ += epsilon

    # Get confidence map
    conf_map = wls_filter.getConfidenceMap()
    conf_map = crop_img(conf_map, 176)

    # Apply bilateral Filter
    bilateral_filtered_displ = cv2.bilateralFilter(displ, d=5, sigmaColor=0.5, sigmaSpace=5)

    # # 원래 disparity 맵 (displ)에서 최소 disparity 값을 가져옵니다.
    # min_disparity = np.min(displ)

    # # bilateral_filtered_displ을 원래 disparity 스케일로 변환합니다.
    # bilateral_filtered_displ = (bilateral_filtered_displ * numDisparities) + min_disparity

    # Apply consistency check
    # h,w = Left_nice.shape[0], Left_nice.shape[1]
    # lr_check = np.zeros((h, w), dtype=np.float32)
    # x, y = np.meshgrid(range(w),range(h))
    # r_x = (x - displ) # x-DL(x,y)
    # print(displ)
    # print(r_x)
    # mask1 = (r_x >= 0) # coordinate should be non-negative integer
    # l_disp = displ[mask1]
    # r_disp = dispr[y[mask1].astype(int), r_x[mask1].astype(int)]
    # mask2 = (l_disp == r_disp) # check if DL(x,y) = DR(x-DL(x,y))
    # lr_check[y[mask1][mask2], x[mask1][mask2]] = displ[mask1][mask2]
    # print(len(lr_check>0))


    return displ, dispr, wls_filtered_displ, conf_map, bilateral_filtered_displ, numDisparities

def get_color_depthmap(depthmap, max_range):
    # 256 단계의 color map을 생성합니다.
    cmap = plt.cm.get_cmap("jet", 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    # sparse depthmap인 경우 depth가 있는 곳만 추출합니다.
    depth_pixel_v_s, depth_pixel_u_s = np.where(depthmap > 0)

    H, W = depthmap.shape
    color_depthmap = np.zeros((H, W, 3)).astype(np.uint8)
    for depth_pixel_v, depth_pixel_u in zip(depth_pixel_v_s, depth_pixel_u_s):
        depth = depthmap[depth_pixel_v, depth_pixel_u]
        color_index = int(255 * min(depth, max_range) / max_range)
        color = cmap[color_index, :]
        cv2.circle(color_depthmap, (depth_pixel_u, depth_pixel_v), 1, color=tuple(color), thickness=-1)
    
    return color_depthmap

def generate_depthmap(Left_nice, displ, wls_filtered_displ, bilateral_filtered_displ, numDisparities):
    ''' (초점거리 * 기준선) / 시차 공식을 사용하여 depthmap 계산하는 함수'''

    mask = displ > 0
    units = 0.512

    depth = (fx_L * baseline) / displ
    # depth = get_color_depthmap(depth, 255)


    # # calculate depth data
    # depth = np.zeros(shape=(Left_nice.shape[0],Left_nice.shape[1])).astype("uint16")
    # depth[mask] = (fx_L * baseline) / (units * displ[mask])
    # print(depth)
    # # visualize depth data
    # depth = cv2.equalizeHist(depth)
    # colorized_depth = np.zeros((Left_nice.shape[0], Left_nice.shape[1], 3), dtype="uint16")
    # temp = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
    # colorized_depth[mask] = temp[mask]
    # # colorized_depth = crop_img(colorized_depth,176)


    plt.imshow(depth)
    plt.show()

    return


def generate_pcd(Left_nice, displ, wls_filtered_displ, bilateral_filtered_displ):
    ''' opencv "reprojectimageto3d"를 사용하여 point cloud 계산하는 함수'''

    # Q matrix = [[1,0,0,-Cx],[0,1,0,-Cy], [0,0,0,f], [0,0,-1/Tx, (Cx-Cx')/Tx]]
    # Cx: 왼쪽 카메라 x 주점
    # Cy: 왼쪽 카메라 y 주점
    # Cx': 오른쪽 카메라 x 주점
    # Tx: 카메라 렌즈 기준선(baseline)

    # Edit Q matrix
    # 범용적인 Q 값
    Q = np.float32([[1, 0, 0, -968.8848659],
                [0, -1, 0, 545.83119],
                [0, 0, 1428.08808, 0],
                [0, 0, 1/60, 1]])

    # 실제 Q 값
    # Q = np.float32([[1, 0, 0, -968.848659],
    #                [0, 1, 0, -545.83119],
    #                [0, 0, 0, 1428.08808],
    #                [0, 0, -1/60, (968.8848659-938.823154)/60]])

    # Q = np.float32([[1/(5.07e-6), 0, 0, -968.8848659],
    #                [0, 1/(3.38e-6), 0, -545.83119],
    #                [0, 0, 0, 1428.08808],
    #                [0, 0, -1/60, (968.8848659-938.823154)/60]])

    # Convert disparity map to point cloud
    no_points = cv2.reprojectImageTo3D(displ, Q)
    wls_points = cv2.reprojectImageTo3D(wls_filtered_displ, Q)
    bilateral_points = cv2.reprojectImageTo3D(bilateral_filtered_displ, Q)

    # wls_points = wls_points[:,:,2].clip(0, np.inf)
    # mask = displ > displ.min()
    mask = displ > 0
    colors = cv2.cvtColor(Left_nice, cv2.COLOR_BGR2RGB)
    
    no_filtered_points = no_points[mask]
    wls_filtered_points = wls_points[mask]
    bilateral_filtered_points = bilateral_points[mask]

    out_colors = colors[mask]
    norm_out_colors = out_colors/255.0

    # Create no filtered point cloud
    point_cloud1 = o3d.geometry.PointCloud()
    point_cloud1.points = o3d.utility.Vector3dVector(no_filtered_points)
    point_cloud1.colors = o3d.utility.Vector3dVector(norm_out_colors)

    # Create wls filtered point cloud
    point_cloud2 = o3d.geometry.PointCloud()
    point_cloud2.points = o3d.utility.Vector3dVector(wls_filtered_points)
    point_cloud2.colors = o3d.utility.Vector3dVector(norm_out_colors)

    # Create wls filtered point cloud
    point_cloud3 = o3d.geometry.PointCloud()
    point_cloud3.points = o3d.utility.Vector3dVector(bilateral_filtered_points)
    point_cloud3.colors = o3d.utility.Vector3dVector(norm_out_colors)

    # Create coordinate frame at 0, 0, 0
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=500, origin=[0, 0, 0])

    # Create a list of geometries
    geometries1 = [point_cloud1, coordinate_frame]
    geometries2 = [point_cloud2, coordinate_frame]
    geometries3 = [point_cloud3, coordinate_frame]
    return wls_points, no_filtered_points, wls_filtered_points, bilateral_filtered_displ, geometries1, geometries2, geometries3


def visualize(Left_nice, Right_nice, displ, wls_filtered_displ, conf_map,bilateral_filtered_displ,  geometries1, geometries2, geometries3):
    '''disparity map, point cloud를 시각화하는 함수'''

    # Visualize point cloud and coordinate frame
    o3d.visualization.draw_geometries(geometries1)
    o3d.visualization.draw_geometries(geometries2)
    o3d.visualization.draw_geometries(geometries3)

    # Crop images
    Left_nice = cv2.cvtColor(Left_nice, cv2.COLOR_BGR2RGB)
    Left_nice = crop_img(Left_nice, 176)
    displ = crop_img(displ, 176)
    wls_filtered_displ = crop_img(wls_filtered_displ, 176)
    bilateral_filtered_displ = crop_img(bilateral_filtered_displ, 176)

    # Visualize disparity and RGB image.
    fig = plt.figure(figsize=(20,20))
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,2,3)
    ax4 = fig.add_subplot(2,2,4)
    ax1.imshow(Left_nice, 'gray')
    ax2.imshow(displ, 'gray')
    ax3.imshow(wls_filtered_displ, 'gray')
    ax4.imshow(bilateral_filtered_displ, 'gray')
    ax1.set_title('RGB Left')
    ax2.set_title('Disparity Left')
    ax3.set_title('WLS_Disparity Left')
    ax4.set_title('Bilateral_Disparity Left')
    plt.show()

    return


def print_coordinates(event, x, y, flags, param):
    ''' opencv window에 마우스 클릭시 해당 마우스 포인터의 픽셀좌표 출력하는 함수'''

    if event == cv2.EVENT_LBUTTONDOWN:  # 마우스 왼쪽 버튼을 눌렀을 때
        print(f'클릭한 픽셀의 좌표는 ({x}, {y}) 입니다.')
    return


def save_pgm(image, file_name, wls_points, output_path):
    ''' 3D point cloud를 입력받아 pgm file을 output 폴더에 저장하는 함수'''

    # Set pgm file name. 
    pgm_name = output_path + '/' + file_name + '.pgm'

    # Return input image size.
    height, width, channel = image.shape

    # Extract Z value from 3D point cloud.
    points = wls_points[:,:,2].astype(int)

    # Convert numpy array to string with aligned digits
    points = '\n'.join([' '.join([str(num).rjust(5) for num in row]) for row in points])

    # Write pgm file
    with open(pgm_name, 'w', newline='') as f:
        f.write('P2\n')
        f.write(f'{width} {height}\n')
        f.write('65535\n')
        f.write(points)
    return

def save_img(image, file_name, output_path):
    ''' rectified image를 output 폴더에 저장하는 함수'''

    # Set img file name.
    img_name = output_path + '/' + file_name + '.png'

    # Crop image
    image = crop_img(image, 176)

    # Save img file in output directory.
    cv2.imwrite(img_name, image)
    return

if __name__=="__main__":
    main()
