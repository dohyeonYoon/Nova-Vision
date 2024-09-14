'''
BRODY v0.1 - Segmentation module

'''

from mmdet.apis import init_detector, inference_detector
import cv2
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import time

def Segment_Broiler(filename, cfg_file, check_file, score_conf):
    """입력 이미지에 Instance Segmentation을 적용하여 육계영역만 분할해주는 함수.

    Args:
        filename: 입력 데이터 이름

    Returns:
        results: 추론 결과 얻어진 bbox, mask 정보
        th_index: confidence score가 threshold값 보다 큰 인스턴스의 index 정보
    """
    # Set image file name.
    img_name = filename + '.png'
    print(img_name)

    # Make Detector model based on config and checkpoint.
    model = init_detector(cfg_file, check_file, device='cuda:0')

    # Read image. 
    img_arr= cv2.imread(img_name, cv2.IMREAD_COLOR)

    # Save inference results(bbox, mask).
    results = inference_detector(model, img_arr)

    # Save index of instance whose conf_score is greater than threshold_value
    th_index = np.where(results[0][0][:,4]> score_conf)
    th_index = th_index[0].tolist()

    return results, th_index


def Get_mask(results, th_index):
    """모든 개체의 2D mask 픽셀좌표 받아오는 함수.

    Args:
        results: Segmentation 결과 얻은 bbox, mask 정보
        th_index: confidence score가 threshold값 보다 큰 인스턴스의 index 정보

    Returns:
        mask_list: 모든 개체의 mask 픽셀좌표 저장된 리스트
    """
    # Save 2D mask pixel points for all objects
    mask_list = []
    for i in th_index:
        mask_array = np.where(results[1][0][i]==1, 255, results[1][0][i]).astype(np.uint8)
        pixels = cv2.findNonZero(mask_array)
        mask_list.append(pixels)
        
    return mask_list


def Get_Contour(results, th_index):
    """Instance Segmentation 결과 얻어진 mask에서 contour 생성해주는 함수.

    Args:
        results: Segmentation 결과 얻은 bbox, mask 정보
        th_index: confidence score가 threshold값 보다 큰 인스턴스의 index 정보

    Returns:
        contour_list (ndarray): 모든 개체의 contour점 픽셀좌표가 저장된 리스트
        th_index: 예외처리 후 나머지 개체의 인덱스가 저장된 리스트
    """
    # Save 2D contour points for all objects
    contour_list = []
    pop_list = []
    for i in th_index:
        # Convert binary mask to array.
        mask_array = np.where(results[1][0][i]==1, 255, results[1][0][i]).astype(np.uint8)

        # Generate contour.
        contour, hierarchy = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_list.append(contour[0])

        # Exception handling1(one object has more than one contour)
        if len(contour) == 1:
            pass
        else:
            pop_list.append(i)
            print(f"{i}번째 인스턴스는 contour가 2개 그려집니다.")

        # Exception handling2(contour have only 1~3 points)
        if len(contour[0]) in [1,3]:
            pop_list.append(i)
            print(f"{i}번째 인스턴스에서 contour point가 1~3개입니다")
        else: 
            pass

    # Apply exception handling.
    pop_list = list(set(pop_list)) # Prevent deduplication
    for k in pop_list:
        th_index.remove(k)

    return contour_list, th_index