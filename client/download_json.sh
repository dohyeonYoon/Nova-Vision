#!/bin/bash

# 설정
SERVER_URL="http://example.com/path/to/config.json"  # B 서버의 JSON 파일 URL
DESTINATION_DIR="/scratch/camera_parm"  # JSON 파일을 저장할 디렉토리
DESTINATION_FILE="$DESTINATION_DIR/config.json"  # JSON 파일 경로

# 디렉토리가 존재하지 않으면 생성
mkdir -p "$DESTINATION_DIR"

# JSON 파일 다운로드
curl -o "$DESTINATION_FILE" "$SERVER_URL"

# 다운로드 성공 여부 확인
if [ $? -eq 0 ]; then
    echo "JSON 파일 다운로드 성공"
else
    echo "JSON 파일 다운로드 실패"
fi

# 사용방법
# 1. download_json.sh 파일을 /usr/local/bin/ 폴더로 옮겨주세요.
# 2. 터미널에서 $ sudo chmod +x /usr/local/bin/download_json.sh 명령어를 실행하여 스크립트 실행권한을 부여해주세요.
# 3. 터미널에서 $ update-rc.d download_json.sh defaults 명령어를 실행하여 부팅 시 자동 실행되도록 설정해주세요.