import numpy as np
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
import os
from natsort import natsorted
from matplotlib import cm, colors

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
    displ, dispr, wls_displ, conf_map, bilateral_displ = compute_disparity(Left_nice, Right_nice, parameter)
    no_depthmap, wls_depthmap, bilateral_depthmap = generate_depthmap(displ, wls_displ, bilateral_displ)
    # generate_pcd(Left_nice, displ, wls_displ, bilateral_displ)
    generate_pcd(Left_nice, displ, wls_displ, bilateral_displ, no_depthmap, wls_depthmap, bilateral_depthmap)
    # visualize(Left_nice, Right_nice, displ, wls_filtered_displ, conf_map,bilateral_filtered_displ, geometries1, geometries2, geometries3)
    # save_pgm(Left_nice, file_name, wls_points, output_path)
    # save_img(Left_nice, file_name, output_path)
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
    displ = left_matcher.compute(Left_nice, Right_nice)
    dispr = right_matcher.compute(Right_nice, Left_nice)

    # Scaling down the disparity values and normalizing them
    epsilon = 1e-7
    norm_displ = (displ.astype(np.float32)/16.0 - minDisparity)/numDisparities
    norm_displ += epsilon
    norm_dispr = (dispr.astype(np.float32)/16.0 - minDisparity)/numDisparities
    norm_dispr += epsilon

    # Apply wls filter to normalized displ
    lmbda = 7000
    sigma = 1.5
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left = left_matcher)
    # wls_filter = cv2.ximgproc.createDisparityWLSFilterGeneric(use_confidence=True)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    wls_displ = wls_filter.filter(displ, Left_nice, None, dispr)
    wls_displ[wls_displ== -32768] = -16 # -32768값을 -16으로 대체

    # Get confidence map
    conf_map = wls_filter.getConfidenceMap()
    conf_map = crop_img(conf_map, 176)

    # Apply bilateral Filter
    bilateral_displ = cv2.bilateralFilter(displ.astype(np.float32), d=5, sigmaColor=0.5, sigmaSpace=5)

    # Divide by 16 (because original disparity value is signed int16 format)
    displ = displ.astype(np.float32) / 16.0 
    wls_displ = wls_displ.astype(np.float32) / 16.0
    bilateral_displ = bilateral_displ.astype(np.float32) / 16.0 

    return displ, dispr, wls_displ, conf_map, bilateral_displ


def generate_depthmap(displ, wls_displ, bilateral_displ):
    
    # Disparity2Depth waveshare company method
    # epsilon = 1e-7
    # min_disp = np.min(displ)
    # displ = displ + (-min_disp)
    # displ = displ / 16.0
    # displ = displ.astype(np.uint8)
    # displ = cv2.normalize(displ, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

    # min_wls_disp = np.min(wls_displ)
    # wls_displ = wls_displ + (-min_wls_disp)
    # wls_displ = wls_displ / 16.0
    # wls_displ = wls_displ.astype(np.uint8)
    # wls_displ = cv2.normalize(wls_displ, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

    # min_bilateral_disp = np.min(bilateral_displ)
    # bilateral_displ = bilateral_displ + (-min_bilateral_disp)
    # bilateral_displ = bilateral_displ / 16.0
    # bilateral_displ = bilateral_displ.astype(np.uint8)
    # bilateral_displ = cv2.normalize(bilateral_displ, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

    print(displ.dtype)
    print(wls_displ.dtype)
    print(bilateral_displ.dtype)
    print(displ.shape)
    print(wls_displ.shape)
    print(bilateral_displ.shape)
    print('min disparity값', np.min(displ))
    print('min disparity값', np.min(wls_displ))
    print('min disparity값', np.min(bilateral_displ))
    print('max disparity값', np.max(displ))
    print('max disparity값', np.max(wls_displ))
    print('max disparity값', np.max(bilateral_displ))

    # Calculate depth
    no_depthmap = fx_L * baseline / displ
    wls_depthmap = fx_L * baseline / wls_displ
    bilateral_depthmap = fx_L * baseline / bilateral_displ
    min_depth = 480.0
    max_depth = 4000.0

    # Dead pixel crop
    # displ = crop_img(displ, 176)
    # wls_displ = crop_img(wls_displ, 176)
    # bilateral_displ = crop_img(bilateral_displ, 176)
    # no_depthmap = crop_img(no_depthmap, 176)
    # wls_depthmap = crop_img(wls_depthmap, 176)
    # bilateral_depthmap = crop_img(bilateral_depthmap, 176)

    # 깊이측정 범위 설정
    no_depthmap = np.where((no_depthmap<min_depth) | (no_depthmap>max_depth), 0, no_depthmap) # depth map의 범위를 0.48~10m로 제한
    wls_depthmap = np.where((wls_depthmap<min_depth) | (wls_depthmap>max_depth), 0, wls_depthmap) # depth map의 범위를 0.48~10m로 제한
    bilateral_depthmap = np.where((bilateral_depthmap<min_depth) | (bilateral_depthmap>max_depth), 0, bilateral_depthmap) # depth map의 범위를 0.48~10m로 제한

    print('no min depth value', np.min(no_depthmap))
    print('wls min depth value', np.min(wls_depthmap))
    print('bilateral min depth value', np.min(bilateral_depthmap))
    print('no max depth value', np.max(no_depthmap))
    print('wls max depth value', np.max(wls_depthmap))
    print('bilateral max depth value', np.max(bilateral_depthmap))

    # 깊이맵 정규화
    # no_dmax = np.max(no_depthmap)
    # wls_dmax = np.max(wls_depthmap)
    # bilateral_dmax = np.max(bilateral_depthmap)
    # normalized_no_depthmap = no_depthmap / no_dmax * 255
    # normalized_wls_depthmap = wls_depthmap / wls_dmax * 255
    # normalized_bilateral_depthmap = bilateral_depthmap / bilateral_dmax * 255

    # normalized_no_depthmap = cv2.normalize(no_depthmap, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    # normalized_wls_depthmap = cv2.normalize(wls_depthmap, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    # normalized_bilateral_depthmap = cv2.normalize(bilateral_depthmap, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

    # Visualize disparity and RGB image.
    # fig = plt.figure(figsize=(20,20))
    # ax1 = fig.add_subplot(2,3,1)
    # ax2 = fig.add_subplot(2,3,2)
    # ax3 = fig.add_subplot(2,3,3)
    # ax4 = fig.add_subplot(2,3,4)
    # ax5 = fig.add_subplot(2,3,5)
    # ax6 = fig.add_subplot(2,3,6)    
    # ax1.imshow(displ, 'gray')
    # ax2.imshow(wls_displ, 'gray')
    # ax3.imshow(bilateral_displ, 'gray')
    # ax4.imshow(normalized_no_depthmap, 'jet')
    # ax5.imshow(normalized_wls_depthmap, 'jet')
    # ax6.imshow(normalized_bilateral_depthmap, 'jet')
    # ax1.set_title('disparity Left')
    # ax2.set_title('wls_disparity Left')
    # ax3.set_title('bilateral_disparity Left')
    # ax4.set_title('no_depth Left')
    # ax5.set_title('wls_depth Left')
    # ax6.set_title('bilateral_depth Left')
    # plt.show()
    
    return no_depthmap, wls_depthmap, bilateral_depthmap


# def generate_pcd(Left_nice, displ, wls_filtered_displ, bilateral_filtered_displ):
#     ''' opencv "reprojectimageto3d"를 사용하여 point cloud 계산하는 함수'''

#     # Q matrix = [[1,0,0,-Cx],[0,1,0,-Cy], [0,0,0,f], [0,0,-1/Tx, (Cx-Cx')/Tx]]
#     # Cx: 왼쪽 카메라 x 주점
#     # Cy: 왼쪽 카메라 y 주점
#     # Cx': 오른쪽 카메라 x 주점
#     # Tx: 카메라 렌즈 기준선(baseline)

#     # Edit Q matrix
#     # 범용적인 Q 값
#     Q = np.float32([[1, 0, 0, 0],
#                 [0, -1, 0, 0],
#                 [0, 0, 1428.08808, 0],
#                 [0, 0, 0, 1]])

#     # 실제 Q 값
#     # Q = np.float32([[1, 0, 0, -968.848659],
#     #                [0, 1, 0, -545.83119],
#     #                [0, 0, 0, 1428.08808],
#     #                [0, 0, -1/60, (968.8848659-938.823154)/60]])

#     # Q = np.float32([[1/(5.07e-6), 0, 0, -968.8848659],
#     #                [0, 1/(3.38e-6), 0, -545.83119],
#     #                [0, 0, 0, 1428.08808],
#     #                [0, 0, -1/60, (968.8848659-938.823154)/60]])

#     # Convert disparity map to point cloud
#     no_points = cv2.reprojectImageTo3D(displ, Q)
#     wls_points = cv2.reprojectImageTo3D(wls_filtered_displ, Q)
#     bilateral_points = cv2.reprojectImageTo3D(bilateral_filtered_displ, Q)

#     # wls_points = wls_points[:,:,2].clip(0, np.inf)
#     # mask = displ > displ.min()
#     mask = displ > 0
#     colors = cv2.cvtColor(Left_nice, cv2.COLOR_BGR2RGB)
    
#     no_filtered_points = no_points[mask]
#     wls_filtered_points = wls_points[mask]
#     bilateral_filtered_points = bilateral_points[mask]

#     out_colors = colors[mask]
#     norm_out_colors = out_colors/255.0

#     # Create no filtered point cloud
#     point_cloud1 = o3d.geometry.PointCloud()
#     point_cloud1.points = o3d.utility.Vector3dVector(no_filtered_points)
#     point_cloud1.colors = o3d.utility.Vector3dVector(norm_out_colors)

#     # Create wls filtered point cloud
#     point_cloud2 = o3d.geometry.PointCloud()
#     point_cloud2.points = o3d.utility.Vector3dVector(wls_filtered_points)
#     point_cloud2.colors = o3d.utility.Vector3dVector(norm_out_colors)

#     # Create wls filtered point cloud
#     point_cloud3 = o3d.geometry.PointCloud()
#     point_cloud3.points = o3d.utility.Vector3dVector(bilateral_filtered_points)
#     point_cloud3.colors = o3d.utility.Vector3dVector(norm_out_colors)

#     # Create coordinate frame at 0, 0, 0
#     coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=500, origin=[0, 0, 0])

#     # Create a list of geometries
#     geometries1 = [point_cloud1, coordinate_frame]
#     geometries2 = [point_cloud2, coordinate_frame]
#     geometries3 = [point_cloud3, coordinate_frame]

#     # Visualize point cloud and coordinate frame
#     o3d.visualization.draw_geometries(geometries1)
#     o3d.visualization.draw_geometries(geometries2)
#     o3d.visualization.draw_geometries(geometries3)

#     return wls_points, no_filtered_points, wls_filtered_points, bilateral_filtered_displ, geometries1, geometries2, geometries3

def generate_pcd(Left_nice, displ, wls_displ, bilateral_displ, no_depthmap, wls_depthmap, bilateral_depthmap):
    ''' depth map을 직접 변환하여 point cloud 계산하는 함수'''

    # Return input depthmap size
    height, width = no_depthmap.shape

    # Save no_depthmap XYZ points
    no_point_cloud = []
    for u in range(height):
        for v in range(width):
            Z = float(no_depthmap[u,v]) # actual 3D z point of corresponding pixel
            Y= ((u-cy_L) * float(Z)) / fy_L # actual 3D y point of corresponding pixel
            X= ((v-cx_L) * float(Z)) / fx_L # actual 3D x point of corresponding pixel
            no_point_cloud.append([X,Y,Z])
    no_point_cloud = np.array(no_point_cloud)
    no_point_cloud = no_point_cloud.reshape(height,width,3)

    # Save wls_depthmap XYZ points
    wls_point_cloud = []
    for u in range(height):
        for v in range(width):
            Z = float(wls_depthmap[u,v]) # actual 3D z point of corresponding pixel
            Y= ((u-cy_L) * float(Z)) / fy_L # actual 3D y point of corresponding pixel
            X= ((v-cx_L) * float(Z)) / fx_L # actual 3D x point of corresponding pixel
            wls_point_cloud.append([X,Y,Z])
    wls_point_cloud = np.array(wls_point_cloud)
    wls_point_cloud = wls_point_cloud.reshape(height,width,3)

    # Save bilateral_depthmap XYZ points
    bilateral_point_cloud = []
    for u in range(height):
        for v in range(width):
            Z = float(bilateral_depthmap[u,v]) # actual 3D z point of corresponding pixel
            Y= ((u-cy_L) * float(Z)) / fy_L # actual 3D y point of corresponding pixel
            X= ((v-cx_L) * float(Z)) / fx_L # actual 3D x point of corresponding pixel
            bilateral_point_cloud.append([X,Y,Z])
    bilateral_point_cloud = np.array(bilateral_point_cloud)
    bilateral_point_cloud = bilateral_point_cloud.reshape(height,width,3)

    # Dead pixel crop
    # displ = crop_img(displ, 176)
    # wls_displ = crop_img(wls_displ, 176)
    # bilateral_displ = crop_img(bilateral_displ, 176)
    # no_depthmap = crop_img(no_depthmap, 176)
    # wls_depthmap = crop_img(wls_depthmap, 176)
    # bilateral_depthmap = crop_img(bilateral_depthmap, 176)

    # Valid pixel
    no_mask = displ > 0
    wls_mask = wls_displ > 0
    bilateral_mask = bilateral_displ > 0
    colors = cv2.cvtColor(Left_nice, cv2.COLOR_BGR2RGB)
    # colors = crop_img(colors, 176)

    no_filtered_points = no_point_cloud[no_mask]
    wls_filtered_points = wls_point_cloud[wls_mask]
    bilateral_filtered_points = bilateral_point_cloud[bilateral_mask]
    no_colors = colors[no_mask]
    wls_colors = colors[wls_mask]
    bilateral_colors = colors[bilateral_mask]
    norm_no_colors = no_colors/255.0
    norm_wls_colors = wls_colors/255.0
    norm_bilateral_colors = bilateral_colors/255.0

    # Create no filtered point cloud
    point_cloud1 = o3d.geometry.PointCloud()
    point_cloud1.points = o3d.utility.Vector3dVector(no_filtered_points)
    point_cloud1.colors = o3d.utility.Vector3dVector(norm_no_colors)

    # Create wls filtered point cloud
    point_cloud2 = o3d.geometry.PointCloud()
    point_cloud2.points = o3d.utility.Vector3dVector(wls_filtered_points)
    point_cloud2.colors = o3d.utility.Vector3dVector(norm_wls_colors)

    # Create wls filtered point cloud
    point_cloud3 = o3d.geometry.PointCloud()
    point_cloud3.points = o3d.utility.Vector3dVector(bilateral_filtered_points)
    point_cloud3.colors = o3d.utility.Vector3dVector(norm_bilateral_colors)

    # Create coordinate frame at 0, 0, 0
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=500, origin=[0, 0, 0])

    # Create a list of geometries
    geometries1 = [point_cloud1, coordinate_frame]
    geometries2 = [point_cloud2, coordinate_frame]
    geometries3 = [point_cloud3, coordinate_frame]

    # Visualize point cloud and coordinate frame
    o3d.visualization.draw_geometries(geometries1)
    o3d.visualization.draw_geometries(geometries2)
    o3d.visualization.draw_geometries(geometries3)

    return


def visualize(Left_nice, Right_nice, displ, wls_filtered_displ, conf_map,bilateral_filtered_displ,  geometries1, geometries2, geometries3):
    '''disparity map, point cloud를 시각화하는 함수'''

    # Visualize point cloud and coordinate frame
    # o3d.visualization.draw_geometries(geometries1)
    # o3d.visualization.draw_geometries(geometries2)
    # o3d.visualization.draw_geometries(geometries3)

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


def print_distance(event, x, y,depth_map, flags, param):
    ''' opencv window에 마우스 클릭시 해당 마우스 포인터의 깊이정보 출력하는 함수'''
    depth_map = param[0]
    if event == cv2.EVENT_LBUTTONDOWN:  # 마우스 왼쪽 버튼을 눌렀을 때
        print(f'클릭한 픽셀의 거리는 {depth_map[y,x]} 입니다.')
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
