# from fastapi import APIRouter, HTTPException
# from pydantic import BaseModel
# import sys

# # pyBRODY.py의 경로를 시스템 경로에 추가
# sys.path.append("/app/ai/weight_prediction")

# from pyBRODY import main

# router = APIRouter()

# class InferenceRequest(BaseModel):
#     data: list

# @router.post("/")
# def inference(request: InferenceRequest):
#     try:
#         result = main()
#         return {"prediction": result}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))




from fastapi import APIRouter, File, UploadFile, HTTPException
import cv2
import numpy as np
from pyBRODY import load_model, predict

router = APIRouter()

# 모델 로드
model = load_model("/api/weight_prediction/model.pth")

@router.post("/inference")
async def inference(file_left: UploadFile = File(...), file_right: UploadFile = File(...)):
    try:
        # 두 개의 이미지를 읽어서 OpenCV로 디코딩
        left_image_bytes = await file_left.read()
        right_image_bytes = await file_right.read()

        left_image = cv2.imdecode(np.frombuffer(left_image_bytes, np.uint8), cv2.IMREAD_COLOR)
        right_image = cv2.imdecode(np.frombuffer(right_image_bytes, np.uint8), cv2.IMREAD_COLOR)

        # 모델 추론 수행
        result = predict([left_image, right_image], model)
        
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

