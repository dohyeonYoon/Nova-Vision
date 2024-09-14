from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import os

router = APIRouter()

CONFIG_DIR = "../camera_conf"

@router.get("/{filename}")
def download_file(filename: str):
    file_path = os.path.join(CONFIG_DIR, filename)
    if os.path.exists(file_path):
        # 추가로, 다운로드 시 파일 이름을 명시적으로 지정
        return FileResponse(
            path=file_path, 
            filename=filename,
            media_type='application/json'  # JSON 파일의 미디어 타입 설정
        )
    else:
        raise HTTPException(status_code=404, detail="File not found")