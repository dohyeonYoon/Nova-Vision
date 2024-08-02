
# 부팅시 camera_config 파일을 자동으로 다운로드하는 스크립트 파일 설정방법
# 1. download_json.sh 파일을 /usr/local/bin/ 폴더로 옮겨주세요.
# 2. 터미널에서 $ sudo chmod +x /usr/local/bin/download_json.sh 명령어를 실행하여 스크립트 실행권한을 부여해주세요.
# 3. 터미널에서 $ update-rc.d download_json.sh defaults 명령어를 실행하여 부팅 시 자동 실행되도록 설정해주세요.