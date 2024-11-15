import cv2
import subprocess
import asyncio
import aiohttp
import json
from datetime import datetime
from aiohttp import ClientTimeout

class StereoCameraStreamer:
    def __init__(self, left_camera_index, right_camera_index, rtmp_url, capture_interval, fastapi_url):
        self.left_camera_index = left_camera_index
        self.right_camera_index = right_camera_index
        self.rtmp_url = rtmp_url
        self.capture_interval = capture_interval
        self.fastapi_url = fastapi_url

        # 스트리밍을 위한 왼쪽 카메라 초기화
        self.left_camera = cv2.VideoCapture(left_camera_index)
        self.left_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.left_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not self.left_camera.isOpened():
            raise ValueError("Error: Cannot open left camera")

    async def stream_video(self):
        ffmpeg_command = [
            'ffmpeg',
            '-f', 'v4l2',
            '-video_size', '640x480',  # 스트리밍 해상도 설정
            '-i', f'/dev/video{self.left_camera_index}',
            '-c:v', 'libx264',
            '-preset', 'veryfast',
            '-f', 'flv',  # RTMP 프로토콜을 위한 포맷
            self.rtmp_url
        ]
        # 비동기적으로 ffmpeg 스트리밍 실행
        process = await asyncio.create_subprocess_exec(*ffmpeg_command)
        await process.communicate()

    async def capture_and_post_images(self):
        while True:
            await asyncio.sleep(self.capture_interval)

            # 캡처를 위한 카메라 설정
            self.left_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.left_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            
            right_camera = cv2.VideoCapture(self.right_camera_index)
            right_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            right_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

            if not right_camera.isOpened():
                print("Error: Cannot open right camera")
                continue

            try:
                # 두 카메라에서 동기화된 프레임 캡처
                self.left_camera.grab()
                right_camera.grab()
                ret_left, frame_left = self.left_camera.retrieve()
                ret_right, frame_right = right_camera.retrieve()

                if not ret_left or not ret_right:
                    print("Error: Failed to capture image")
                    continue

                success_left, buffer_left = cv2.imencode('.png', frame_left)
                success_right, buffer_right = cv2.imencode('.png', frame_right)

                if not success_left or not success_right:
                    print("Error: Failed to encode image")
                    continue

                image_data_left = buffer_left.tobytes()
                image_data_right = buffer_right.tobytes()

                # FastAPI 서버에 POST 요청 비동기로 전송
                await asyncio.gather(
                    self.post_image(image_data_left, "left"),
                    self.post_image(image_data_right, "right")
                )
            finally:
                right_camera.release()

    async def post_image(self, image_data, side):
        url = f"{self.fastapi_url}/predict"
        timeout = ClientTimeout(total=60) # 타임아웃 시간 설정
        
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                files = {'file': image_data}
                data = {'side': side}
                headers = {'Content-Type': 'image/png'}

                async with session.post(url, data=data, headers=headers) as response:
                    if response.status == 200:
                        print(f"{side} image posted successfully")
                    else:
                        print(f"Failed to post {side} image with status code: {response.status}")
        except Exception as e:
            print(f"HTTP 전송 실패 from {side} camera: {e}")

    async def start(self):
        # stream_video와 capture_and_post_images 비동기적으로 동시에 실행
        await asyncio.gather(
            self.stream_video(),
            self.capture_and_post_images()
        )

    def release(self):
        self.left_camera.release()

# 설정 파일 읽기
with open('config.json', 'r') as f:
    config = json.load(f)

fastapi_url = config['fastapi_url']
rtmp_url = config['rtmp_url']
capture_interval = config['capture_interval']
left_camera_index = config['left_camera_index']
right_camera_index = config['right_camera_index']

# 스트리머 인스턴스 생성 및 실행
streamer = StereoCameraStreamer(left_camera_index, right_camera_index, rtmp_url, capture_interval, fastapi_url)
try:
    asyncio.run(streamer.start())
finally:
    streamer.release()
