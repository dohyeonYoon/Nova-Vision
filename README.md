# Nova-Vision

AI기반 가축 성장관리 통합 솔루션 **Nova-Vision**입니다.
Nova-Vision은 스테레오 카메라로 수집된 3D 데이터를 이용하여 가축의 생장정보(표면적, 부피, 체장 등)를 계산하고 이를 기반으로 체중을 예측합니다. 이를 통해 농장주는 가축이 잘 자라고 있는지 실시간으로 모니터링하고 최적의 출하시기를 결정할 수 있습니다.
https://github.com/user-attachments/assets/57aa3c0a-33d1-475f-a1db-6494a6175607



## 🎯 Technical issues & Resolution process

* [Nova-Vision은 어떻게 개발했을까?: 3D Depth camera 개발](https://dohyeon.tistory.com/73)
* [Nova-Vision은 어떻게 개발했을까?: AI 가축 체중측정 알고리즘 개발](https://dohyeon.tistory.com/95)


## :heavy_check_mark: Tested

| Python |  Windows   |   Mac   |   Linux  |
| :----: | :--------: | :-----: | :------: |
| 3.8.0+ | Windows 10 | X |  X |


## :arrow_down: Installation

Clone repo and install [requirements.txt](https://github.com/dohyeonYoon/pyStereo/blob/main/requirements.txt) in a
**Python>=3.8.0** environment, including


```bash
git clone https://github.com/dohyeonYoon/pyStereo  # clone
cd pyStereo
pip install -r requirements.txt  # dependency install
```


## :blue_book: Process

1. Print the checkerboard PDF file and use the stereo_capture.py code to capture images simultaneously from both cameras (recommended to capture more than 50 images).
2. Use the captured checkerboard images and the calibration.py code to calibrate both cameras.
3. Adjust the parameters of the disparity map using the disparity.py code.
4. Reconstruct the disparity map using the img2disp.py code and reconstruct a 3D point cloud from the disparity map.


## :rocket: Getting started

You can inference with your own custom left & right image in ./data/checkboard_10x7/stereoL, ./data/checkboard_10x7/stereoR folder.
```bash
python img2disp.py

```


### :file_folder: stereo_rectify parameter & dataset 
Please place the downloaded **stereo_rectify_map.xml** file in /data directory and **dataset** files in /data/checkboard_10x7/stereoL, stereoR directory respectively.

[stereo_rectify_map](https://drive.google.com/file/d/1QBbd0ebVYuPQontHv6U8a9jx2eZXnTzA/view?usp=sharing)  # stereo_rectify_map parameter

[dataset](https://drive.google.com/drive/folders/1DCtE4_Gq5DGBjRF43g7JsRbamI510pI2?usp=sharing)  # Datasets
