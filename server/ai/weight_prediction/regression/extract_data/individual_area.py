'''
BRODY v0.1 - Calculate_individual_broiler_area module

'''

import cv2
import os
import numpy as np 
from shapely.geometry import Polygon
from natsort import natsorted
from math import sqrt
from csv import writer
from datetime import datetime

# 경로선언
date_list = ['2022-04-28','2022-04-30','2022-05-02','2022-05-04',
            '2022-05-06','2022-05-08','2022-05-10','2022-05-12',
            '2022-05-14','2022-05-16','2022-05-18','2022-05-20',
            '2022-05-22','2022-05-24','2022-05-26','2022-05-28',
            '2022-05-30']
arrival_date = [2022,4,26,00]

def main():
    for i in date_list:
        origin_img_path = f'C:/Users/MSDL-DESK-02/Desktop/data_arrange/input/{i}/SegmentationClass'
        binary_img_path = f'C:/Users/MSDL-DESK-02/Desktop/data_arrange/output/{i}/'
        origin_depthmap_path = f'C:/Users/MSDL-DESK-02/Desktop/data_arrange/pgm/{i}'
        img_file_list = natsorted(os.listdir(origin_img_path))
        binary_file_list = natsorted(os.listdir(binary_img_path))
        depthmap_file_list = natsorted(os.listdir(origin_depthmap_path))

        Mask2binary(origin_img_path, img_file_list, binary_img_path)
        days = Calculate_day_function(i, arrival_date)
        contour_list, extream_point_list, centroid_list = Get_contour(binary_img_path, binary_file_list)
        array_3d, contour_list_3d = Convert_2D_to_3D(origin_img_path, img_file_list, origin_depthmap_path, depthmap_file_list, centroid_list, contour_list)
        area_list, perimeter_list = Calculate_2d_area(contour_list_3d)
        major_axis_list, minor_axis_list = Calculate_major_minor_axis(extream_point_list, array_3d)
        Save_to_csv(days, area_list, perimeter_list, major_axis_list, minor_axis_list)
    
    return

def Mask2binary(origin_img_path, img_file_list, binary_img_path):
    """annotation된 이미지를 바이너리 이미지로 변환한뒤 저장해주는 함수.

    Args:
        origin_img_path: 촬영 날짜
        arrival_date: 병아리가 들어온 일자(1일령일 때 날짜)

    Returns:
        days: 현재날짜의 육계군집의 일령(육계의 나이는 1일령,5일령...등으로 표현).
    """
    for i in range(len(img_file_list)):
        img_name = origin_img_path + '/' + img_file_list[i]
        img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        ret, img_binary = cv2.threshold(img, 1,255, cv2.THRESH_BINARY)
        cv2.imwrite(binary_img_path +f'{i}.png' , img_binary)
    return

def Calculate_day_function(date, arrival_date):
    """촬영당시 육계의 일령 계산해주는 함수.

    Args:
        date: 촬영 날짜
        arrival_date: 병아리가 들어온 일자(1일령일 때 날짜)

    Returns:
        days: 현재날짜의 육계군집의 일령(육계의 나이는 1일령,5일령...등으로 표현).
    """
    date_value = date.split('-')
    date_value = date_value[0:]

    # 촬영날짜 및 병아리 입식날짜
    dt1 = datetime(int(date_value[0]),int(date_value[1]),int(date_value[2]),10)
    dt2 = datetime(arrival_date[0],arrival_date[1],arrival_date[2],arrival_date[3]) # 육계가 1일령일 때 날짜를 입력해줄 것(시간은 00시 기준). 

    # 일령계산
    td = dt1-dt2
    days = td.days + 1

    return days

def Get_contour(binary_img_path, binary_file_list):
    contour_list = []
    extream_point_list = []
    centroid_list = []
    for i in range(len(binary_file_list)):
        # 바이너리 이미지 불러오기.
        binary_img_name = binary_img_path + '/' + binary_file_list[i]
        binary_img = cv2.imread(binary_img_name)
        img_gray = cv2.cvtColor(binary_img, cv2.COLOR_BGR2GRAY)

        # contour 생성.
        contour, hierarchy = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_list.append(contour) # contour_list에 contour 좌표 append
        cnt = contour[0]

        # contour의 극점 추출(상하좌우)
        topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
        bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
        leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
        rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
        extream_point_list.append([topmost, bottommost, leftmost, rightmost])

        # contour의 무게중심점 추출
        M= cv2.moments(contour)
        centroid_list.append([int(M['m10']/M['m00']), int(M['m01']/M['m00'])])


        contour_list = np.array(contour_list)
        centroid_list = np.array(centroid_list)
    return contour_list, extream_point_list, centroid_list

def Convert_2D_to_3D(origin_img_path, img_file_list, origin_depthmap_path, depthmap_file_list, centroid_list, contour_list):
    for i in range(len(img_file_list)):
        # 입력 이미지 사이즈 반환.
        img_name = origin_img_path + '/' + img_file_list[i]
        height, width, channel = cv2.imread(img_name).shape

        # Depth 정보 parsing.
        depthmap_name = origin_depthmap_path + '/' + depthmap_file_list[i]
        depth_list = []
        with open(depthmap_name, 'r') as f:
            data = f.readlines()[3:]
            for i in data:
                for j in i.split():
                    depth_list.append(int(j))

        # depth map을 이미지 형태(height*width)로 reshape.
        depth_map = np.array(depth_list)
        depth_map = np.reshape(depth_map, (height,width))

        # 무게중심에 median filter 기능 추가
        windows = [0,0,0,0,0,0,0,0,0]
        median_array= depth_map.copy()

        for i in centroid_list:
            windows[0] = depth_map[i[1]-1, i[0]-1] # height, width 순서
            windows[1] = depth_map[i[1]-1, i[0]]
            windows[2] = depth_map[i[1]-1, i[0]+1]
            windows[3] = depth_map[i[1], i[0]-1]
            windows[4] = depth_map[i[1], i[0]]
            windows[5] = depth_map[i[1], i[0]+1]
            windows[6] = depth_map[i[1]+1, i[0]-1]
            windows[7] = depth_map[i[1]+1, i[0]]
            windows[8] = depth_map[i[1]+1, i[0]+1]
            windows = np.array(windows)
            windows.sort()
            median_array[i[1], i[0]] = windows[4]


        # median filter가 적용된 무게중심점의 z값을 contour에 뿌리기
        z_c_list = []
        for j in range(len(centroid_list)):
            z_c_list.append(median_array[centroid_list[j][1],centroid_list[j][0]])
            for k in contour_list[j][0]:
                median_array[k[0][1],k[0][0]] = int(median_array[centroid_list[j][1],centroid_list[j][0]])
            median_array = np.array(median_array)

        # 카메라 내부 파라미터를 이용한 3차원 변환
        fx=535.99
        fy=536.16
        cx=638.31
        cy=367.2195
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

        # 각 instance의 contour points 3차원 좌표 list에 저장.
        contour_list_3d = []
        for i in range(len(contour_list)):
            point_list0 = []
            for j in range(len(contour_list[i][0])): # i번째 육계의 contour 개수
                points = array_3d[contour_list[i][0][j][0][1], contour_list[i][0][j][0][0]] # height, width 순서 
                point_list0.append(points)
            contour_list_3d.append(point_list0)
        contour_list_3d = np.array(contour_list_3d, dtype= object)
        
    return array_3d, contour_list_3d

def Calculate_2d_area(contour_list_3d):
    # 면적 및 둘레 계산
    area_list = []
    perimeter_list = []
    for i in range(len(contour_list_3d)):
        polygon = Polygon(contour_list_3d[i])
        polygon_area = round((polygon.area)/100,2)
        polygon_perimeter = round((polygon.length)/10,2)
        area_list.append(polygon_area)
        perimeter_list.append(polygon_perimeter)

    return area_list, perimeter_list

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


def Save_to_csv(days, area_list, perimeter_list, major_axis_list, minor_axis_list):
    # 일령, 면적, 둘레, 장축, 단축 5가지를 csv file에 저장.
    with open('C:/Users/MSDL-DESK-02/Desktop/data_arrange/individual_data.csv','a', newline='') as f_object:
        writer_object = writer(f_object)
        for i in range(len(area_list)):
            rows = [days, area_list[i], perimeter_list[i], major_axis_list[i], minor_axis_list[i]]
            writer_object.writerow(rows)
        f_object.close()

    return

if __name__ == "__main__":
    main()