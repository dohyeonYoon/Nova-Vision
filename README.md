<div align="center">
  <br>
  <picture>
    <source srcset="./docs/imgs/nova-vision_logo.png" media="(prefers-color-scheme: dark)">
    <img width="370" src="./docs/imgs/nova-vision_logo.png">
  </picture>
  
  <h2>농장주를 위한 가축 성장관리 서비스</h2></hr>
  <p align="center">
    <img src="https://img.shields.io/badge/fastapi-%23009688.svg?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI badge">
    <img src="https://img.shields.io/badge/react-%2361DAFB.svg?style=for-the-badge&logo=react&logoColor=black" alt="React badge">
    <img src="https://img.shields.io/badge/pytorch-%23EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch badge">
    <img src="https://img.shields.io/badge/opencv-%235C3EE8.svg?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV badge">
    <img src="https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white" alt="Docker badge">
    <img src="https://img.shields.io/badge/timescaledb-%23364F6E.svg?style=for-the-badge&logo=timescale&logoColor=white" alt="TimescaleDB badge">    <img src="https://img.shields.io/badge/aws-%23FF9900.svg?style=for-the-badge&logo=amazonaws&logoColor=white" alt="AWS badge">
    <img src="https://img.shields.io/badge/open3d-%23000000.svg?style=for-the-badge&logo=open3d&logoColor=white" alt="Open3D badge">
    <img src="https://img.shields.io/badge/mmdetection-%23E95420.svg?style=for-the-badge&logo=github&logoColor=white" alt="MMDetection badge">
  </p>
</div>
<br>

## 프로젝트 설명

<strong>Nova-Vision</strong>은 이미지만으로 가축의 체중을 측정하여 가축의 성장관리를 도와주는 서비스입니다. 🐓  
3D 카메라로 수집한 이미지로 가축의 체중을 측정하고 성장률을 시각화하여 농장주에게 최적의 출하시기를 알려줍니다.


<div align="center">
<img width="45%" src="./docs/imgs/example1.png"><img width="45%" src="./docs/imgs/example2.png">
</div>

## 🎯 Technical issues & Resolution process

* [Nova-Vision은 어떻게 개발했을까?: 3D Depth camera 개발](https://dohyeon.tistory.com/73)
* [Nova-Vision은 어떻게 개발했을까?: AI 가축 체중측정 알고리즘 개발](https://dohyeon.tistory.com/95)
* [Nova-Vision은 어떻게 개발했을까?: 모델 경량화 및 코드 최적화](https://dohyeon.tistory.com/107)

## 홍보 영상

<div align="center">
    <a href="https://youtu.be/kOnUdoRliY8">
        <img src="https://github.com/user-attachments/assets/203b0488-93c1-4ab5-b3ff-18c4db5054a7" alt="Video Label" width="60%">
    </a>
</div>

## 프로젝트 아키텍쳐

### 서비스 아키텍쳐

<img width="85%" src="./docs/imgs/architecture_service5.png"/>

### 모델 아키텍쳐

<img width="100%" src="./docs/imgs/architecture_model.png"/>

## Features

### 실시간 영상 스트리밍 및 캡처

> 실시간 영상 스트리밍과 이미지 캡처를 동시에!
<img width="80%" src="./docs/imgs/example3.png"/>

24시간 영상을 스트리밍하면서 한 시간에 한 번씩 이미지를 캡처해서 서버로 전송합니다.


<br/>

### Depth Estimation 

> 오직 두 장의 이미지로 3D 데이터 획득!
![example4](https://github.com/user-attachments/assets/f7fe48f2-4029-461f-9114-8e371dd37ed7)

카메라로부터 전송받은 두 장의 이미지만으로 3D 데이터(point cloud)를 복원합니다.


<br/>

### 가축 체중측정

> 이미지만으로 실시간으로 가축 체중측정!
<img width="90%" src="./docs/imgs/example5.png"/>
복원된 3D 데이터를 활용하여 가축의 생체지표(부피,표면적 등)를 계산하고 체중을 측정합니다.


<br/>

### 가축 성장관리

> 24시간 가축의 성장상태 분석 및 모니터링!
<img width="80%" src="./docs/imgs/example6.png"/>

농장주가 가축의 성장상태를 한눈에 파악할 수 있도록 다양한 지표들을 시각화합니다. 
<br/>
