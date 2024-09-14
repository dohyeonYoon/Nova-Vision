'''
BRODY v0.1 - Archive module

'''

import numpy as np
import os
from shapely.geometry import Polygon, mapping, shape
from natsort import natsorted
from csv import writer
from datetime import datetime
import joblib
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import cv2
import matplotlib.pyplot as plt
from pprint import pprint
import time
import pickle
import random
import open3d as o3d
import alphashape
from sklearn.preprocessing import MinMaxScaler
import time
from math import sqrt

# ------------------- 2D Area ------------------- #

def Calculate_major_minor_axis(extream_point_list, array_3d):
    """개별 개체의 장축,단축 길이 계산해주는 함수.

    Args:
        extream_point_list: 모든 개체의 상하좌우 극점이 저장된 리스트
        array_3d: 모든 픽셀의 3차원 좌표가 저장된 array

    Returns:
        major_axis_list: 모든 개체의 major_axis 길이가 저장된 리스트
        minor_axis_list: 모든 개체의 minor_axis 길이가 저장된 리스트
    """
    # extream point를 잇는 major, minor axis length 계산
    major_axis_list = []
    minor_axis_list = []

    for i in range(len(extream_point_list)):
        tm_2d = list(extream_point_list[i][0]) # 상 2D
        bm_2d = list(extream_point_list[i][1]) # 하 2D
        lm_2d = list(extream_point_list[i][2]) # 좌 2D
        rm_2d = list(extream_point_list[i][3]) # 우 2D

        # 상하좌우 극점의 3D 좌표
        tm_3d = list(array_3d[tm_2d[1], tm_2d[0]]) # 상 3D
        bm_3d = list(array_3d[bm_2d[1], bm_2d[0]]) # 하 3D
        lm_3d = list(array_3d[lm_2d[1], lm_2d[0]]) # 좌 3D
        rm_3d = list(array_3d[rm_2d[1], rm_2d[0]]) # 우 3D

        # 상-하, 좌-우 3D 극점 사이의 거리
        distance_tb = sqrt((tm_3d[0]-bm_3d[0])**2 + (tm_3d[1]-bm_3d[1])**2 + (tm_3d[2]-bm_3d[2])**2)
        distance_lr = sqrt((lm_3d[0]-rm_3d[0])**2 + (lm_3d[1]-rm_3d[1])**2 + (lm_3d[2]-rm_3d[2])**2) 

        # 더 큰 값을 major_axis_list에 삽입
        bigger_distance = round(max(distance_tb, distance_lr),2)
        smaller_distnace = round(min(distance_tb, distance_lr),2)
        major_axis_list.append(bigger_distance)
        minor_axis_list.append(smaller_distnace)

    return major_axis_list, minor_axis_list

def Calculate_perpendicular_point(array_3d):
    w,h = array_3d.shape[0], array_3d.shape[1]
    total = w*h 
    array3 = array_3d.reshape(total,3)

    # 포인트 클라우드 정의
    pcd_plane = o3d.geometry.PointCloud()
    pcd_plane.points = o3d.utility.Vector3dVector(array3)
    pcd_plane.normals = o3d.utility.Vector3dVector(array3)

    # 지면 법선벡터 계산 및 flip 
    pcd_plane.estimate_normals()
    pcd_plane.orient_normals_towards_camera_location(pcd_plane.get_center())
    pcd_plane.normals = o3d.utility.Vector3dVector( - np.asarray(pcd_plane.normals))

    # 지면 평면 계산 및 시각화
    plane_model, inliers = pcd_plane.segment_plane(distance_threshold=0.1,
                                        ransac_n=3,
                                        num_iterations=1000)
    [a, b, c, camera_height] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {camera_height:.2f} = 0")  
    camera_height = - camera_height

    perpendicular_point = (0,0,camera_height)

    return perpendicular_point

def Calculate_distance(array_3d, centroid_list, perpendicular_point, exclude_index_list):
    """각 instance의 무게중심점의 3차원 좌표를 추출하고, 수선의발에서부터 각 instance까지 거리 구하는 함수.

    """

    # 각 instance의 무게중심점 3차원 좌표 list에 저장.
    center_of_mass_list = []
    exclude_centroid_list = []

    for i in exclude_index_list:
        exclude_centroid_list.append(centroid_list[i])

    for i in range(len(exclude_centroid_list)): # 육계 개체 수
        point_x, point_y = array_3d[exclude_centroid_list[i][1], exclude_centroid_list[i][0]][0],array_3d[exclude_centroid_list[i][1], exclude_centroid_list[i][0]][1]  # height, width 순서 
        center_of_mass_list.append([point_x,point_y])
    
    # 유클리디안 거리 계산
    perpendicular_point = [0,0]
    distance_list = []

    for i in center_of_mass_list:
        distance = sqrt((perpendicular_point[0]-i[0])**2 +(perpendicular_point[1]-i[1])**2)
        distance_list.append(distance)
    
    return distance_list

def Find_straight_line(contour_list, exclude_depth_err_index_list):
    """ 기둥에 가려진(segmented mask가 직선을 갖는) 개체의 인덱스 찾아주는 함수.  
    Args:
        contour_list: 모든 개체의 contour점 픽셀좌표가 저장된 리스트
        exclude_depth_err_index_list: 깊이 이상치 개체를 제외한 나머지 개체 인덱스가 저장된 리스트

    Returns:
        straight_line_index_list: 기둥에 가려진(segmentated mask가 직선을 갖는) 개체의 인덱스가 저장된 리스트
    """
    straight_line_index_list = []
    for i in exclude_depth_err_index_list:
        for j in range(len(contour_list[i][0]-1)):
            dist = cv2.norm(contour_list[i][0][j], contour_list[i][0][j+1])
            if dist >= 35:
                print(f"{i}번째 인스턴스에 35픽셀 이상의 직선이 있습니다.")
                straight_line_index_list.append(i)
            else:
                pass

    return straight_line_index_list

def Find_pillar(img_name):
    """ 이미지에서 기둥 영역 찾아주는 함수.  
    Args:
        img_name: RGB(.png) 파일 디렉토리 경로

    Returns:
        pillar: 원본 이미지에서 기둥만 분할된 바이너리 이미지
    """
    # HSV color space로 이미지 불러들이기
    img = cv2.imread(img_name)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 물체 분할을 위한 HSV 범위 설정
    lower_range = np.array([10, 5, 22])
    upper_range = np.array([231, 30, 67])

    # 설정된 HSV 범위에 해당하는 물체 분할
    mask = cv2.inRange(hsv, lower_range, upper_range)

    # 물체 분할을 위해 원본 이미지에 bitwise_and 적용
    pillar = cv2.bitwise_and(img, img, mask=mask)

    return pillar

def Min_max_scale(days_list, area_list, perimeter_list, major_axis_list):
    max_day = 35
    max_area = 404.62
    max_perimeter = 106.15
    max_major_axis = 347.39 
    min_day = 1
    min_area = 28.71
    min_perimeter = 20.94
    min_major_axis = 66.61

    scaled_days_list = []
    scaled_area_list = []
    scaled_perimeter_list = []
    scaled_major_axis_list = []

    for i in days_list:
        new_days_x = (i - min_day) / (max_day - min_day)
        scaled_days_list.append(new_days_x)

    for j in area_list:
        new_area_x = (j - min_area) / (max_area - min_area)
        scaled_area_list.append(new_area_x)

    for k in perimeter_list:
        new_perimeter_x = (k - min_perimeter) / (max_perimeter - min_perimeter)
        scaled_perimeter_list.append(new_perimeter_x)

    for m in perimeter_list:
        new_major_axis_x = (m - min_major_axis) / (max_major_axis - min_major_axis)
        scaled_major_axis_list.append(new_major_axis_x)

    return scaled_days_list, scaled_area_list, scaled_perimeter_list, scaled_major_axis_list

# ------------------- 3D Surface ------------------- #

def Convert_2D_to_3D_surface(pgm_file_path, mask_list):
    """depth map(.pgm) file을 불러와서 각 instance를 이루는 모든 픽셀좌표를, 3차원 월드좌표로 변환해주는 함수.

    Args:
        pgm_file_path: depth map(.pgm) file을 불러오는 디렉토리 경로.
        mask_list: instance를 이루는 모든 픽셀좌표가 저장된 list

    Returns:
        list_converted_to_3d: 
    """

    # 입력 이미지 사이즈 반환.
    height, width, channel = cv2.imread(img_name).shape

    # Depth 정보 parsing.
    depth_list = []
    with open(depthmap_name, 'r') as f:
        data = f.readlines()[3:]
        for i in data:
            for j in i.split():
                depth_list.append(int(j))

    # depth map을 이미지 형태(height*width)로 reshape.
    depth_map = np.array(depth_list)
    depth_map = np.reshape(depth_map, (height,width))

    # 카메라 내부 파라미터를 이용한 3차원 월드좌표(실제좌표)로 변환.
    fx = 535.14 # 초점거리 x
    fy = 535.325 # 초점거리 y
    cx= 646.415 # 주점 x 
    cy= 361.3215 # 주점 y
    factor = 1.0

    # array_3d 리스트에 X,Y,Z 좌표 저장
    array_3d = []
    for u in range(height):
        for v in range(width):
            Z = float(median_array[u,v]) / factor # 해당 픽셀의 3차원상의 실제좌표 z
            Y= ((u-cy) * float(Z)) / fy # 해당 픽셀의 3차원상의 실제좌표 y
            X= ((v-cx) * float(Z)) / fx # 해당 픽셀의 3차원상의 실제좌표 x
            array_3d.append([X,Y,Z])
    array_3d = np.array(array_3d)
    array_3d = array_3d.reshape(height,width,3)

    # 각 instance의 mask points 3차원 좌표 list에 저장.
    mask_list_3d = []
    for i in mask_list:
        point_list0 = []
        for j in range(len(i)): # i번째 육계의 pixel 개수
            points = array_3d[i[j][0][1], i[j][0][0]] # height, width 순서 
            point_list0.append(points)
        mask_list_3d.append(point_list0)

    # 각 instance mask의 3차원 좌표 numpy array로 변환.
    mask_list_3d = np.array(mask_list_3d, dtype= object)

    return mask_list_3d, array_3d

def Remove_outlier(mask_list_3d):

    # Before remove outlier points
    for i in mask_list_3d:
        fig = plt.figure()
        ax = fig.gca(projection = '3d')
        X = i[:,0]
        Y = i[:,1]
        Z = i[:,2]

        x,y = np.meshgrid(X,Y)
        plane = 0.001*x + 0.001*y + 2400
        # ax.plot_surface(x,y,plane)
        ax.scatter(X,Y,Z)
        ax.set_xlabel("x coordinate")
        ax.set_ylabel("y coordinate")
        ax.set_zlabel("z coordinate")
        plt.suptitle("point cloud of mask", fontsize=16)
        plt.gca().invert_xaxis()
        plt.gca().invert_zaxis()
        plt.show()

    # After remove outlier points
    filtered_mask_point_list = []
    for i in range(len(mask_list_3d)):
        mask_point = o3d.geometry.PointCloud()
        mask_point.points = o3d.utility.Vector3dVector(mask_list_3d[i])
        mask_point.normals = o3d.utility.Vector3dVector(mask_list_3d[i])

        filtered_mask_point,ind = mask_point.remove_statistical_outlier(nb_neighbors=60, std_ratio=0.8)
        filtered_mask_point = np.asarray(filtered_mask_point.points)
        filtered_mask_point_list.append(filtered_mask_point)

        fig = plt.figure()
        ax = fig.gca(projection = '3d')
        X = filtered_mask_point_list[i][:,0]
        Y = filtered_mask_point_list[i][:,1]
        Z = filtered_mask_point_list[i][:,2]

        x,y = np.meshgrid(X,Y)
        plane = 0.001*x + 0.001*y + 2400
        # ax.plot_surface(x,y,plane)
        ax.scatter(X,Y,Z)
        ax.set_xlabel("x coordinate")
        ax.set_ylabel("y coordinate")
        ax.set_zlabel("z coordinate")
        plt.suptitle("point cloud of mask", fontsize=16)
        plt.gca().invert_xaxis()
        plt.gca().invert_zaxis()
        plt.show()

    return filtered_mask_point_list

def Project_mesh_to_plane(filtered_mask_point_list):
    projected_mask_list = []
    
    # 각 instance를 이루는 point들의 z좌표를 0으로 변환(=지면에 투영)
    for i in filtered_mask_point_list:
        projected_mask_list0 = []
        projected_mask_list1 = []
        for j in range(len(i)):
            projected_mask_list0.append(i[j][0])
            projected_mask_list0.append(i[j][1])

        # 2개씩 묶어주기
        for j in range(0,len(projected_mask_list0),2):
            projected_mask_list1.append(tuple(projected_mask_list0[j:j+2]))
        projected_mask_list.append(projected_mask_list1)

    return projected_mask_list

def Find_boundary_points(projected_mask_list):
    start = time.time()
    projected_boundary_points = []
    for i in range(len(projected_mask_list)):
        # 각 instance를 이루는 point cloud list 입력
        xy =  np.array(projected_mask_list[i])

        # minmax scaler 범위 조정
        scaler = MinMaxScaler(feature_range = (0.0,0.01)) # feature 범위를 0~1사이로 변환 
        scaler.fit(xy)
        transformed_xy = scaler.transform(xy)

        # 범위 조정된 points x,y 좌표
        transformed_x, transformed_y = transformed_xy[:, 0], transformed_xy[:, 1]

        # 알파 쉐이프 형성
        shapes = alphashape.alphashape(transformed_xy, alpha=100.0)
        shapes_x, shapes_y = shapes.exterior.coords.xy

        shapes_xy0 = []
        shapes_xy = []
        for i in range(len(shapes_x)):
            shapes_xy0.append(shapes_x[i])
            shapes_xy0.append(shapes_y[i])
        for j in range(0,len(shapes_xy0),2):
            shapes_xy.append(tuple(shapes_xy0[j:j+2]))

        # 원래 범위로 inverse 변환
        inverse_transformed_xy = scaler.inverse_transform(transformed_xy)
        inverse_transformed_shapes_xy = scaler.inverse_transform(shapes_xy)

        x ,y = inverse_transformed_xy[:, 0], inverse_transformed_xy[:, 1]
        shape_x, shape_y = inverse_transformed_shapes_xy[:,0], inverse_transformed_shapes_xy[:,1]
    
        # # 시각화
        plt.plot(x, y, 'o', color='black', markersize=6)
        plt.plot(shape_x, shape_y, 'o', color='red', markersize=4)
        plt.gca().invert_yaxis()
        plt.show()

        projected_boundary_points.append(inverse_transformed_shapes_xy)
    end = time.time()

    # print("boundary points 계산 코드동작 시간은", end-start, "입니다.")
    return projected_boundary_points

def Calculate_projected_area(projected_boundary_points):
    # shapely Polygon 함수로 면적 계산
    area_list = []
    for i in range(len(projected_boundary_points)):
        polygon = Polygon(np.array(projected_boundary_points[i]))
        polygon_area = round((polygon.area)/100,2)
        area_list.append(polygon_area)
    average_area = int((sum(area_list,0.0)/ len(area_list)))
    return area_list, average_area

def Calculate_camera_height(array2):
    array3 = array2.reshape(921600,3)

    # 포인트 클라우드 정의
    pcd_plane = o3d.geometry.PointCloud()
    pcd_plane.points = o3d.utility.Vector3dVector(array3)
    pcd_plane.normals = o3d.utility.Vector3dVector(array3)

    # 지면 법선벡터 계산 및 flip 
    pcd_plane.estimate_normals()
    pcd_plane.orient_normals_towards_camera_location(pcd_plane.get_center())
    pcd_plane.normals = o3d.utility.Vector3dVector( - np.asarray(pcd_plane.normals))

    # 지면 평면 계산 및 시각화
    plane_model, inliers = pcd_plane.segment_plane(distance_threshold=0.1,
                                        ransac_n=3,
                                        num_iterations=1000)
    [a, b, c, camera_height] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {camera_height:.2f} = 0")  
    camera_height = - camera_height
    return camera_height

def Calculate_angle_between(filtered_mask_point_list):

    # 각 instance의 point cloud의 무게중심점 list 생성
    center_of_gravity_list0 = []
    center_of_gravity_list = []
    for i in filtered_mask_point_list:
        center_x = sum(i[:,0]) / len(i[:,0])
        center_y = sum(i[:,1]) / len(i[:,1])
        center_z = sum(i[:,2]) / len(i[:,2])
        center_of_gravity_list0.append(center_x)
        center_of_gravity_list0.append(center_y)
        center_of_gravity_list0.append(center_z)
    
    for j in range(0,len(center_of_gravity_list0),3):
        center_of_gravity_list.append(center_of_gravity_list0[j:j+3])
    
    # 초점에서 무게중심점을 이은 벡터를 단위벡터로 변환
    unit_vector_instance_list = []
    for i in center_of_gravity_list:
        unit_vector_instance_list.append(i / np.linalg.norm(i)) # center_of_gravity에 저장된 모든 벡터를 단위벡터로 변환

    unit_vector_vertical = np.array([0.0,0.0, 1.0]) # 초점에서 지면 중심점까지 내린 수선의 발 단위벡터
    unit_vector_instance_list = np.array(unit_vector_instance_list)

    # 두 단위 벡터 사이의 각도 반환
    pi = 22/7
    angle_list = []
    for i in unit_vector_instance_list: # 두 단위벡터 사이각 계산
        angle_radian = np.arccos(np.clip(np.dot(unit_vector_vertical, i), -1.0, 1.0))
        angle_degree = angle_radian*(180/pi) # 라디안을 각도단위로 변환
        angle_list.append(angle_degree) # 계산된 모든 각도를 angle_list에 삽임


    angle_list = np.array(angle_list)

    return angle_list

def Calculate_correct_value(area_list, keys):
    avg_area = {"2022_04_26":34.58,"2022_04_28":42.79,"2022_04_30":50.80,"2022_05_02":57.55,
                "2022_05_04":69.46,"2022_05_06":85.78,"2022_05_08":97.08,"2022_05_10":112.58,
                "2022_05_12":127.76,"2022_05_14":131.03,"2022_05_16":176.51,"2022_05_18":185.71,
                "2022_05_20":201.80,"2022_05_22":223.28,"2022_05_24":210.26,"2022_05_26":276.80,
                "2022_05_28":297.29,"2022_05_30":346.92}
    correct_value_list = []
    for i in area_list:
        k = avg_area[f"{keys}"]/ i
        correct_value_list.append(k)
    print("보정상수의 개수는? ",len(correct_value_list))
    return correct_value_list

def Predict_weight(area_list):
    predict_weight_list = []
    model = joblib.load('/scratch/dohyeon/mmdetection/linear_model/projection_algorithm_model/Linear.pkl')

    # 선형회귀 네트워크로 체중예측한 결과를 list에 저장.
    for i in area_list:
        predict_weight = model.predict([[i]])
        predict_weight_list.append(predict_weight)
    # print(len(predict_weight_list))
    # print(predict_weight_list)

    # 예측 평균체중, 마리수 변수에 저장.
    predict_average_weight = int((sum(predict_weight_list,0.0)/ len(predict_weight_list)))
    print("평균체중은 ",predict_average_weight, "g 입니다.")
    min_weight = min(predict_weight_list)
    max_weight = max(predict_weight_list)
    print("min weight는", min(predict_weight_list))
    print("max weight는", max(predict_weight_list))

    num_chicken = len(predict_weight_list)

    return predict_weight_list, predict_average_weight, num_chicken, min_weight, max_weight

def Generate_mesh(surface_save_path, mask_list_3d):
    """ 3차원 변환된 mask정보를 이용해 mesh생성 및 각 mesh마다 법선벡터를 생성해주는 함수.

    Args:
        surface_save_path: surface(.vtk) file을 저장해주는 경로
        mask_list_3d: 각 instance를 이루는 모든 3D 월드좌표가 저장된 list

    Returns:
        normals: 각 instance를 이루는 모든 mesh의 법선벡터 값이 저장된 list 
    """
    normals_list = []
    for i in range(len(mask_list_3d)): # 탐지된 모든 개체를 이루는 3D point mesh 및 법선벡터 생성
        cloud = pv.PolyData(mask_list_3d[i].tolist())
        surf = cloud.delaunay_2d()
        surf.compute_normals(cell_normals=True, point_normals=False, flip_normals=True, inplace=True) # cell normal = 면 법선, point_normals = 점 법선
        normals = surf.active_normals
        normals = np.array(normals)
        normals_list.append(normals)
  
    normals_list = np.array(normals_list)
    
    return normals_list

def Convert_to_unit_vector(normals_list): 
    """ mesh를 이루는 법선벡터를 단위벡터로 변환해주는 함수.  
    Args:
        normals: 각 instance를 이루는 모든 mesh의 법선벡터 값이 저장된 list

    Returns:
        unit_vector_normals: 각 instance를 이루는 모든 mesh의 법선벡터 값을 단위벡터로 변환한 값을 저장한 list
    """

    unit_vector_normals = []
    for i in normals_list:
        unit_vector_normals0 = [] # 중간저장 list
        for j in i:
            unit_vector_normals0.append(j / np.linalg.norm(j)) # normals list에 저장된 모든 법선벡터를 단위벡터로 변환
        unit_vector_normals.append(unit_vector_normals0)

    unit_vector_normals = np.array(unit_vector_normals)

    return unit_vector_normals


def Calculate_angle_between(unit_vector_normals):
    """두 단위벡터 사이의 각도를 반환해주는 함수.

    Args:
        unit_vector_normals: 각 instance를 이루는 모든 mesh의 법선벡터 값을 단위벡터로 변환한 값을 저장한 list

    Returns:
        angle_list: 영상중심점 벡터와 모든 개체의 무게중심점 벡터 사이의 각도를 저장한 list

    """
    unit_vector_vertical = np.array([0.0,0.0, -1.0]) #가상의 평면에서 카메라 중심방향으로 그린 수직 단위벡터
    unit_vector_normals = np.array(unit_vector_normals)

    pi = 22/7

    
    angle_list = []
    for i in unit_vector_normals: # 두 단위벡터 사이각 계산
        angle_list0 = [] # 중간저장 list
        for j in i:
            angle_radian = np.arccos(np.clip(np.dot(unit_vector_vertical, j), -1.0, 1.0))
            angle_degree = angle_radian*(180/pi) # 라디안을 각도단위로 변환
            angle_list0.append(angle_degree) # 계산된 모든 각도를 angle_list에 삽임
        angle_list.append(angle_list0)

    angle_list = np.array(angle_list)

    # 기준벡터와 이루는 각도가 60도 이상인 cell의 인덱스
    remove_cell_list0=[]
    remove_cell_list = []
    for i in angle_list:
        i = np.array(i)
        remove_cell_index = np.where(i>90)
        remove_cell_list.append(remove_cell_index)

    print(remove_cell_list[0][0])

    f = open("/scratch/dohyeon/mmdetection/output/remove.csv",'w')
    for i in remove_cell_list[0][0]:
        f.write(f"{i}"+" ")

    f.close()

    return remove_cell_list

def Calculate_area_3d(mask_list_3d, remove_cell_list):

    for i in range(len(mask_list_3d)): # 탐지된 모든 개체를 이루는 3D point mesh 및 법선벡터 생성
        cloud = pv.PolyData(mask_list_3d[i].tolist())
        surf = cloud.delaunay_2d()
        edit_surf = surf.remove_cells(remove_cell_list[i])
        edit_surface_area = edit_surf.area

    return


def Exclude_weight_outlying(predict_weight_list, predict_average_weight, inner_object_list):
    # 체중 이상치 제거(+- 30%)
    edit_inner_object_list = []
    edit_predict_weight_list = []
    for index, value in enumerate(predict_weight_list):
        if value < predict_average_weight * 0.7 or value > predict_average_weight * 1.3:
            # 배제하도록 pass
            pass
        else:
            edit_inner_object_list.append(inner_object_list[index])
            edit_predict_weight_list.append(value)
    edit_predict_average_weight = int(sum(edit_predict_weight_list,0.0)/ len(edit_predict_weight_list))

    # print("edit_inner_object_list 개수는", len(edit_inner_object_list))
    # print("edit_predict_weight_list 개수는", len(edit_inner_object_list))
    # print("edit_inner_object_list는", edit_inner_object_list)
    edit_num_chicken = len(predict_weight_list)


    return edit_inner_object_list, edit_predict_weight_list, edit_predict_average_weight, edit_num_chicken

def Visualize_weight(input_path,
                    output_path,
                    results, 
                    predict_weight_list, 
                    date, 
                    exclude_index_list, 
                    predict_average_weight, 
                    days,
                    num_chicken,
                    real_z_c_list,
                    average_area, 
                    score_threshold,
                    area_list,
                    rgb_file_name,):
    """각 개체마다 id number, 예측체중 시각화해주는 함수.

    Args:
        input_path: 입력 이미지를 불러오는 디렉토리 경로.
        output_path: 입력 이미지 위에 segmentation 결과가 그려진 이미지가 저장되는 디렉토리 경로
        reesults: segment_chicken 함수의 반환값(Instance segmentation 결과값 배열)
        area_list: 예측면적 저장된 리스트
        date: 파일이름에서 추출한 촬영시각
        z_c_list: 모든 개체의 무게중심점의 z값을 저장한 list
        
    Returns:
        None
    """
    
    # config 파일을 설정하고, 학습한 checkpoint file 불러오기.
    # config_file = '/scratch/dohyeon/mmdetection/custom_config/num_dataset_15.py' # Mask-RCNN-Dataset_15
    # checkpoint_file = '/scratch/dohyeon/mmdetection/weights/mask_rcnn_r101/num_dataset_15/epoch_36.pth' # Mask-RCNN-Dataset_15
    config_file = '/scratch/dohyeon/mmdetection/custom_config/num_dataset_30.py' # Mask-RCNN-Dataset_30
    checkpoint_file = '/scratch/dohyeon/mmdetection/weights/mask_rcnn_r101/num_dataset_30/epoch_36.pth' # Mask-RCNN-Dataset_30
    # config_file = '/scratch/dohyeon/mmdetection/custom_config/num_dataset_68.py' # Mask-RCNN-Dataset_68
    # checkpoint_file = '/scratch/dohyeon/mmdetection/weights/mask_rcnn_r101/num_dataset_68/epoch_35.pth' # Mask-RCNN-Dataset_68
    # config_file = '/scratch/dohyeon/mmdetection/custom_config/num_dataset_87.py' # Mask-RCNN-Dataset_87
    # checkpoint_file = '/scratch/dohyeon/mmdetection/weights/mask_rcnn_r101/num_dataset_87/epoch_35.pth' # Mask-RCNN-Dataset_87

    # config 파일과 checkpoint를 기반으로 Detector 모델을 생성.
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    
    # 경로선언.
    path_dir = input_path
    save_dir1 = output_path
    file_list = natsorted(os.listdir(path_dir))
    
    # 입력 이미지에 추론결과 visualize
    for i in range(1): 
        img_name = path_dir + '/' + rgb_file_name
        img_arr= cv2.imread(img_name, cv2.IMREAD_COLOR)
        img_arr_rgb = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        fig= plt.figure(figsize=(12, 12))
        plt.imshow(img_arr_rgb)

        # 추론결과 디렉토리에 저장(confidenece score 0.7이상의 instance만 이미지에 그릴 것).
        model.show_result(img_arr,
                        results,
                        predict_weight_list,
                        date,
                        days,
                        num_chicken,
                        average_area,
                        exclude_index_list,
                        predict_average_weight,
                        area_list,
                        real_z_c_list,
                        score_thr=score_threshold,
                        bbox_color=(0,0,0),
                        thickness=0.01,
                        font_size=8,
                        out_file= f'{save_dir1}{rgb_file_name}')
    return


def Visualize_depthmap(depth_map):
    # depthmap 시각화
    max_range = 255
    scaler = MinMaxScaler(feature_range = (0,255)) # feature 범위를 0~1사이로 변환 
    scaler.fit(depth_map)
    depthmap = scaler.transform(depth_map)
    
    cmap = plt.cm.get_cmap("gray", 256)
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
    plt.imshow(color_depthmap)
    plt.show() 
    
    return


def Find_Proj_Plane(mask_list_3d):

    for mask in mask_list_3d:
        # Compute the center of mass
        center_of_mass = np.mean(mask, axis=0)

        # Translate center of mass point at the origin
        centered_mask = mask - center_of_mass

        # Compute the covariance matrix
        covariance_matrix = np.cov(centered_mask, rowvar=False)

        # Compute the eigenvectors from covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Normalize eigenvectors
        normalized_eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0)

        # Construct the rotation matrix from the normalized eigenvectors
        rotation_matrix = normalized_eigenvectors.T

        # Compute normal_vector from rotation matrix
        normal_vector = rotation_matrix[:,2]

        # Define the plane using the normal vector and a point(center of mass point) on the plane
        d = -np.dot(normal_vector, center_of_mass)
        plane_params = np.append(normal_vector, d)

        print(plane_params)

        # Project the 3D points onto the plane
        projected_points = (np.dot(plane_params, mask.T) / np.linalg.norm(normal_vector)).T

        return