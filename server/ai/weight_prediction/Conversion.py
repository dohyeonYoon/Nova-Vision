'''
BRODY v0.1 - Convert module

'''

import numpy as np
from shapely.geometry import Polygon, mapping, shape
import joblib
import cv2
from math import sqrt

# def Generate_Depthmap_1(filename, contour_list, th_index):
#     """깊이정보를 받아와서 각 개체의 무게중심점에 3x3 median filter를 적용하고
#        무게중심점의 z값을 contour에 뿌려주는 함수.

#     Args:
#         filename: 입력 데이터 이름
#         contour_list (ndarray): 모든 개체의 contour점 픽셀좌표가 저장된 리스트
#         th_index: 예외처리 후 나머지 개체의 인덱스가 저장된 리스트

#     Returns:
#         array_3d: 모든 픽셀의 3차원 좌표가 저장된 array
#         th_index: 나비모양, 끊긴모양 등 예외처리 후 나머지 개체의 인덱스가 저장된 리스트
#     """
#     # Set image,depthmap file name.
#     img_name = filename + '.png'
#     depth_name = filename + '.pgm'

#     # Return input image size.
#     height, width, channel = cv2.imread(img_name).shape

#     # Depth data parsing.
#     depth_list = []
#     with open(depth_name, 'r') as f:
#         lines = f.readlines()
#         if lines[0] == 'P2\n' and lines[1] == f'{width} {height}\n':
#             for i in lines[3:]:
#                 for j in i.split():
#                     depth_list.append(int(j))
#         else:
#             print("depthmap file이 pgm ascii 형식이 아니거나 RGB 이미지 크기와 다릅니다.")

#     # Reshape depthmap into image shape.
#     depth_map = np.array(depth_list)
#     depth_map = np.reshape(depth_map, (height,width))

#     # Calculate centroid points of all objects(+butterfly shape and broken shape exceptions)
#     ctr_list = []
#     pop_list = []
#     for i in th_index:
#         M= cv2.moments(contour_list[i])
#         if M["m00"] != 0:
#             ctr_list.append([int(M['m10']/M['m00']), int(M['m01']/M['m00'])])

#         elif M["m00"] == 0:
#             print(f"{i}번째 인스턴스에서 비정상 contour가 발생하였습니다")
#             ctr_list.append([0, 0])
#             pop_list.append(i)

#         else:
#             pass
#     ctr_list = np.array(ctr_list)

#     # Apply exception handling.
#     pop_list = list(set(pop_list)) # Prevent deduplication
#     for k in pop_list:
#         th_index.remove(k)

#     # Apply 3x3 size median filter to centroid points.
#     for i in ctr_list:
#         depth_map[i[1], i[0]] = np.median(depth_map[i[1]-1:i[1]+2, i[0]-1:i[0]+2])

#     # Distribute z value of centroid point to the contour.
#     for j in range(len(ctr_list)):
#         for k in contour_list[j]:
#             depth_map[k[0][1],k[0][0]] = int(depth_map[ctr_list[j][1],ctr_list[j][0]])

#     return depth_map, th_index



def Generate_Depthmap_1(filename, contour_list, th_index):
    """깊이정보를 받아와서 각 개체의 contour에 3x3 median filter를 적용해주는 함수.

    Args:
        filename: 입력 데이터 이름
        contour_list (ndarray): 모든 개체의 contour점 픽셀좌표가 저장된 리스트

    Returns:
        array_3d: 모든 픽셀의 3차원 좌표가 저장된 array
    """
    # Set image,depthmap file name.
    img_name = filename + '.png'
    depth_name = filename + '.pgm'

    # Return input image size.
    height, width, channel = cv2.imread(img_name).shape

    # Depth data parsing.
    depth_list = []
    with open(depth_name, 'r') as f:
        lines = f.readlines()
        if lines[0] == 'P2\n' and lines[1] == f'{width} {height}\n':
            for i in lines[3:]:
                for j in i.split():
                    depth_list.append(int(j))
        else:
            print("depthmap file이 pgm ascii 형식이 아니거나 RGB 이미지 크기와 다릅니다.")

    # Reshape depthmap into image shape.
    depth_map = np.array(depth_list)
    depth_map = np.reshape(depth_map, (height,width))

    # # Apply 3x3 size median filter to contour points.
    # for i in contour_list:
    #     for j in i:
    #         depth_map[j[0][1], j[0][0]] = np.median(depth_map[j[0][1]-1:j[0][1]+2, j[0][0]-1:j[0][0]+2])

    return depth_map


def Convert_3D(filename, depth_map, mask_list):
    """2차원 픽셀좌표계를 3차원 월드좌표계로 변환해주는 함수.

    Args:
        filename: 입력 데이터 이름
        depth_map: 모든 픽셀의 깊이정보(z값)가 저장된 array

    Returns:
        array_3d: 모든 픽셀의 3차원 좌표가 저장된 array
    """
    # Set image,depthmap file name.
    img_name = filename + '.png'

    # Return input image size.
    height, width, channel = cv2.imread(img_name).shape

    # Convert to 3D world coordinates using camera internal parameters
    fx = 535.14 # focal length x
    fy = 535.325 # focal length y
    cx= 646.415 # principal point x
    cy= 361.3215 # principal point y

    # Save XYZ points of each object in array_3d
    array_3d = []
    for u in range(height):
        for v in range(width):
            Z = float(depth_map[u,v]) # actual 3D z point of corresponding pixel
            Y= ((u-cy) * float(Z)) / fy # actual 3D y point of corresponding pixel
            X= ((v-cx) * float(Z)) / fx # actual 3D x point of corresponding pixel
            array_3d.append([X,Y,Z])
    array_3d = np.array(array_3d)
    array_3d = array_3d.reshape(height,width,3)
    # print("변경 전!!!!",array_3d)
    # array_3d = array_3d.tolist()
    # print("변경 후!!!!",array_3d)

    # Save XYZ points of each object mask in mask_list_3d.
    mask_list_3d = []
    # print(mask_list)
    for i in mask_list:
        point_list = []
        for j in i: # i번째 육계의 pixel 개수
            point = array_3d[j[0][1], j[0][0]] # height, width 순서 
            point_list.append(point)
        mask_list_3d.append(point_list)

    # mask_list_3d = [[array_3d[j[0][1], j[0][0]] for j in i] for i in mask_list]
    # mask_list_3d = np.array(mask_list_3d, dtype= object)
    # print(mask_list_3d.shape)
    # print(mask_list_3d)

    return array_3d, mask_list_3d