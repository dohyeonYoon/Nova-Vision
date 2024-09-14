from fastapi import FastAPI
from routes import download, inference

app = FastAPI()

# 라우터 등록
app.include_router(download.router, prefix="/download", tags=["Download"])
app.include_router(inference.router, prefix="/inference", tags=["Inference"])