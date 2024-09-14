'''
BRODY v0.1 - Visualization module

'''

# from mmdet.apis import init_detector
from csv import writer
from datetime import datetime
from natsort import natsorted
from mmdet.apis import init_detector
import os 
import cv2
import numpy as np

def Calculate_Day(filename, start_date):
    """촬영당시 육계의 일령 계산해주는 함수.

    Args:
        filename: 입력 데이터 이름
        start_date: 병아리가 들어온 일자(1일령일 때 날짜)

    Returns:
        days: 현재날짜의 육계군집의 일령(육계의 나이는 1일령,5일령...등으로 표현).
    """
    # Extract date from filename.
    filename = os.path.split(filename)[1]
    date_value = filename.split('_')
    date_value = date_value[0:]

    # Capture date and broiler stocking date
    dt1 = datetime(int(date_value[0]),int(date_value[1]),int(date_value[2]),int(date_value[3]))
    dt2 = datetime(start_date[0],start_date[1],start_date[2],start_date[3]) # 육계가 1일령일 때 날짜를 입력해줄 것(시간은 00시 기준).

    # Calculate days
    td = dt1-dt2
    days = td.days + 1

    return days


def Build_PNG(filename,
              start_date,
              results,
              area_list,
              weight_list, 
              th_index,
              cfg_file,
              check_file):
    """이미지 내의 각 개체마다 id, 체중 시각화해주는 함수.

    Args:
        filename: 입력 데이터 이름
        start_date: 병아리가 들어온 일자(1일령일 때 날짜)
        results: 추론 결과 얻어진 bbox, mask 정보
        area_list: 모든 개체의 면적이 저장된 리스트
        weight_list: 모든 개체의 예측체중이 저장된 리스트
        th_index: 깊이 이상치 개체를 제외한 나머지 개체 인덱스가 저장된 리스트
        
    Returns:
        None
    """

    # Make Detector model based on config and checkpoint.
    model = init_detector(cfg_file, check_file, device='cuda:0')
    
    # Calculate days
    days = Calculate_Day(filename, start_date)

    # Calculate average area, weight.
    average_area = np.round(sum(area_list,0.0)/ len(area_list),2)
    average_weight = np.round((sum(weight_list,0.0)/ len(weight_list)), 2)
    print(f"평균 {average_weight}g입니다")

    # Set image file name.
    img_name = filename + '.png'
    img_arr= cv2.imread(img_name, cv2.IMREAD_COLOR)

    # Extract filename from path 
    filename = os.path.split(filename)[1]
    
    # Visualize inference result.
    model.show_result(img_arr,
                    results,
                    area_list,
                    weight_list,
                    th_index,
                    days,
                    average_area,
                    average_weight,
                    score_thr=0.7,
                    bbox_color=(0,0,0),
                    text_color=(255, 255, 255),
                    thickness=0.01,
                    font_size=12,
                    out_file= f'./output/{filename}.png')

    return


def Save_CSV(filename, start_date, area_list, weight_list):
    """각 개체의 일령, 면적, 체중을 csv file에 저장해주는 함수.

    Args:
        filename: 입력 데이터 이름
        days: 일령
        area_list: 각 instance의 면적이 저장된 리스트
        predict_weight_list: 각 instance의 예측체중이 저장된 리스트

    Returns:
        None
    """
    # Calculate days
    days = Calculate_Day(filename, start_date)

    # Extract filename from path
    filename = os.path.split(filename)[1]

    # # Generate label(days, area, weight)
    # rows = ['days', 'area', 'weight']
    # with open(f'./output/result.csv','a', newline='') as f_object:
    #     # using csv.writer method from CSV package
    #     writer_object = writer(f_object)
    #     writer_object.writerow(rows)
    #     f_object.close()

    avg_area = sum(area_list) / len(area_list)
    avg_weight = sum(weight_list) / len(weight_list) 

    # Save to csv file(days, area, weight).
    for i in range(len(area_list)):
        rows = [days, area_list[i], weight_list[i]]
        with open(f'./output/output.csv','a', newline='') as f_object:
            # using csv.writer method from CSV package
            writer_object = writer(f_object)
            writer_object.writerow(rows)
            f_object.close()

    # # Save to csv file(days, avg_area, avg_weight).
    # rows = [days, avg_area, avg_weight]
    # with open(f'./output/result.csv','a', newline='') as f_object:
    #     # using csv.writer method from CSV package
    #     writer_object = writer(f_object)
    #     writer_object.writerow(rows)
    #     f_object.close()

    return
