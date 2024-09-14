# Nginx와 RTMP 모듈을 포함한 Dockerfile

FROM alfg/nginx-rtmp:latest

# Nginx 설정 파일을 복사합니다
COPY nginx.conf /etc/nginx/nginx.conf

# 추가로 필요한 파일이나 디렉토리를 복사합니다 (예: SSL 인증서 등)

EXPOSE 1935 8080

CMD ["nginx", "-g", "daemon off;"]