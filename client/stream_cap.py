import cv2
import subprocess
import threading
import time
from datetime import datetime
from ftplib import FTP
import io
import json

class StereoCameraStreamer:
    def __init__(self, left_camera_index, right_camera_index, rtmp_url, capture_interval, ftp_details):
        self.left_camera_index = left_camera_index
        self.right_camera_index = right_camera_index
        self.rtmp_url = rtmp_url
        self.capture_interval = capture_interval
        self.ftp_details = ftp_details
        self.lock = threading.Lock()
        
        # 스트리밍을 위한 왼쪽 카메라 초기화
        self.left_camera = cv2.VideoCapture(left_camera_index)
        self.left_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.left_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not self.left_camera.isOpened():
            raise ValueError("Error: Cannot open left camera")
        
        self.stream_thread = threading.Thread(target=self.stream_video)
        self.capture_thread = threading.Thread(target=self.capture_and_transfer_images)
    
    def stream_video(self):
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
        subprocess.run(ffmpeg_command)

    def capture_and_transfer_images(self):
        while True:
            time.sleep(self.capture_interval)
            with self.lock:
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
                    
                    self.transfer_image_ftp(image_data_left, "left")
                    self.transfer_image_ftp(image_data_right, "right")
                finally:
                    right_camera.release()

    def transfer_image_ftp(self, image_data, side):
        attempts = 3  # 재시도 횟수 설정
        for attempt in range(attempts):
            try:
                with FTP(self.ftp_details['hostname']) as ftp:
                    ftp.login(self.ftp_details['username'], self.ftp_details['password'])
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    remote_image_path = f"{self.ftp_details['remote_path']}/capture_{side}_{timestamp}.png"
                    with io.BytesIO(image_data) as bio:
                        ftp.storbinary(f"STOR {remote_image_path}", bio)
                break  # 전송 성공 시 루프 종료
            except Exception as e:
                print(f"FTP 전송 실패 from {side} camera (시도 {attempt + 1}/{attempts}): {e}")
                if attempt < attempts - 1:
                    time.sleep(5)  # 재시도 전 대기
                else:
                    print(f"FTP 전송 포기 from {side} camera")

    def start(self):
        self.stream_thread.start()
        self.capture_thread.start()
    
    def join(self):
        self.stream_thread.join()
        self.capture_thread.join()
    
    def release(self):
        self.left_camera.release()

# 설정 파일 읽기
with open('config.json', 'r') as f:
    config = json.load(f)

ftp_details = config['ftp_details']
rtmp_url = config['rtmp_url']
capture_interval = config['capture_interval']
left_camera_index = config['left_camera_index']
right_camera_index = config['right_camera_index']

try:
    streamer = StereoCameraStreamer(left_camera_index, right_camera_index, rtmp_url, capture_interval, ftp_details)
    streamer.start()
    streamer.join()
finally:
    streamer.release()