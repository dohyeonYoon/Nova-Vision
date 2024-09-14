import os
import cv2
import numpy as np
import open3d as o3d

class StereoDepthEstimator:
    def __init__(self, stereo_rectify_file, depth_param_file, fx_L, baseline):
        self.cv_file = cv2.FileStorage(stereo_rectify_file, cv2.FILE_STORAGE_READ)
        self.parameter = cv2.FileStorage(depth_param_file, cv2.FILE_STORAGE_READ)
        
        # Camera parameters
        self.fx_L = fx_L
        self.baseline = baseline

    def rectify_images(self, imgL, imgR):
        Left_Stereo_Map_x = self.cv_file.getNode("Left_Stereo_Map_x").mat()
        Left_Stereo_Map_y = self.cv_file.getNode("Left_Stereo_Map_y").mat()
        Right_Stereo_Map_x = self.cv_file.getNode("Right_Stereo_Map_x").mat()
        Right_Stereo_Map_y = self.cv_file.getNode("Right_Stereo_Map_y").mat()
        
        imgL_rectified = cv2.remap(imgL, Left_Stereo_Map_x, Left_Stereo_Map_y, cv2.INTER_LANCZOS4)
        imgR_rectified = cv2.remap(imgR, Right_Stereo_Map_x, Right_Stereo_Map_y, cv2.INTER_LANCZOS4)
        
        return imgL_rectified, imgR_rectified
    
    def compute_disparity(self, imgL, imgR):
        # Load StereoSGBM parameters
        minDisparity = int(self.parameter.getNode("minDisparity").real())
        numDisparities = int(self.parameter.getNode("numDisparities").real())
        blockSize = int(self.parameter.getNode("blockSize").real())
        P1 = int(self.parameter.getNode("P1").real())
        P2 = int(self.parameter.getNode("P2").real())
        disp12MaxDiff = int(self.parameter.getNode("disp12MaxDiff").real())
        preFilterCap = int(self.parameter.getNode("preFilterCap").real())
        uniquenessRatio = int(self.parameter.getNode("uniquenessRatio").real())
        speckleWindowSize = int(self.parameter.getNode("speckleWindowSize").real())
        speckleRange = int(self.parameter.getNode("speckleRange").real())
        lmbda = 7000
        sigma = 1.5

        # Create StereoSGBM and WLS filter
        left_matcher = cv2.StereoSGBM_create(
            minDisparity=minDisparity,
            numDisparities=numDisparities,
            blockSize=blockSize,
            P1=P1,
            P2=P2,
            disp12MaxDiff=disp12MaxDiff,
            preFilterCap=preFilterCap,
            uniquenessRatio=uniquenessRatio,
            speckleWindowSize=speckleWindowSize,
            speckleRange=speckleRange,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
        wls_filter.setLambda(lmbda)
        wls_filter.setSigmaColor(sigma)
        
        # Compute disparity
        displ = left_matcher.compute(imgL, imgR).astype(np.float32) / 16.0
        dispr = right_matcher.compute(imgR, imgL).astype(np.float32) / 16.0
        wls_displ = wls_filter.filter(displ, imgL, None, dispr)

        # Crop disparity maps
        displ = self.crop_img(displ, numDisparities)
        wls_displ = self.crop_img(wls_displ, numDisparities)
        
        return displ, wls_displ

    def crop_img(self, array, numDisparities):
        return array[:, numDisparities:]

    def generate_depth_map(self, disparity_map):
        depth_map = self.fx_L * self.baseline / (disparity_map + 1e-7)
        min_depth, max_depth = 480.0, 4000.0
        depth_map = np.clip(depth_map, min_depth, max_depth)
        return depth_map

    def process(self, imgL, imgR):
        imgL_rectified, imgR_rectified = self.rectify_images(imgL, imgR)
        displ, wls_displ = self.compute_disparity(imgL_rectified, imgR_rectified)
        depth_map = self.generate_depth_map(wls_displ)
        return depth_map

# 외부에서 쉽게 사용할 수 있도록 함수 추가
def get_depth_map(imgL, imgR, stereo_rectify_file, depth_param_file, fx_L, baseline):
    depth_estimator = StereoDepthEstimator(stereo_rectify_file, depth_param_file, fx_L, baseline)
    return depth_estimator.process(imgL, imgR)