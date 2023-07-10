# pyStereo
This code is for custom stereo depth estimation.


## :heavy_check_mark: Tested

| Python |  Windows   |   Mac   |   Linux  |
| :----: | :--------: | :-----: | :------: |
| 3.8.0+ | Windows 10 | X |  X |



## :arrow_down: Installation
1. Print the checkerboard PDF file and use the stereo_capture.py code to capture images simultaneously from both cameras (recommended to capture more than 50 images).
2. Use the captured checkerboard images and the calibration.py code to calibrate both cameras.
3. Adjust the parameters of the disparity map using the disparity.py code.
4. Reconstruct the disparity map using the img2disp.py code and reconstruct a 3D point cloud from the disparity map.

Clone repo and install [requirements.txt](https://github.com/dohyeonYoon/pyStereo/blob/main/requirements.txt) in a
**Python>=3.8.0** environment, including


```bash
git clone https://github.com/dohyeonYoon/pyStereo  # clone
cd pyStereo
pip install -r requirements.txt  # dependency install
```

## :blue_book: Process

## :rocket: Getting started

You can inference with your own custom left & right image in ./data/checkboard_10x7/stereoL, ./data/checkboard_10x7/stereoR folder.
```bash
python img2disp.py

```


### :file_folder: stereo_rectify parameter & dataset 
Please place the downloaded **stereo_rectify_map.xml** file in /data directory and **dataset** files in /data/checkboard_10x7/stereoL, stereoR directory respectively.

[stereo_rectify_map](https://drive.google.com/file/d/1QBbd0ebVYuPQontHv6U8a9jx2eZXnTzA/view?usp=sharing)  # stereo_rectify_map parameter

[dataset](https://drive.google.com/drive/folders/1DCtE4_Gq5DGBjRF43g7JsRbamI510pI2?usp=sharing)  # Datasets
