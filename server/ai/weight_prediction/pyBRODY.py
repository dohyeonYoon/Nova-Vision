'''
BRODY v0.1 - Main module

'''

import Segmentation as Seg
import Conversion as Conv
import Exclusion as Exclu
import Projection as Proj
import Prediction as Pred
import Visualization as Visual

from natsort import natsorted
import os

# Set broiler stocking date
start_date = [2022, 4, 26, 00]

# Set input and output path
input_path = './input' 
output_path = './output'
input_file_list = natsorted(os.listdir(input_path))
filename_list = natsorted(list(set(os.path.splitext(i)[0] for i in input_file_list)))

# Config and checkpoint for MMDetection
# cfg_file = './mmdetection/config/mask_rcnn_r101_fpn_n_dataset_30.py'
# check_file = './mmdetection/weights/mask_rcnn_r101_fpn_epoch36_data30.pth'
cfg_file = './mmdetection/config/mask_rcnn_r101_fpn_n_dataset_87.py'
check_file = './mmdetection/weights/mask_rcnn_r101_fpn_epoch35_data87.pth'

score_conf = 0.7

def main():
    for filename in filename_list:
        filename = input_path + '/' + filename

        # Segmentation
        results, th_index = Seg.Segment_Broiler(filename, cfg_file, check_file, score_conf)
        mask_list = Seg.Get_mask(results, th_index)
        contour_list, th_index = Seg.Get_Contour(results, th_index)
        
        # # Conversion
        depth_map = Conv.Generate_Depthmap_1(filename, contour_list, th_index)
        array_3d, mask_list_3d = Conv.Convert_3D(filename, depth_map, mask_list)

        # Exclusion
        filtered_mask_list_3d = Exclu.Remove_outlier(mask_list_3d)
        th_index = Exclu.Delete_Exterior(filename, contour_list, th_index)
        th_index = Exclu.Delete_Depth_Error(contour_list, array_3d, th_index)

        # Projection
        area_list, polygon_type_list = Proj.Find_Proj_Plane(filtered_mask_list_3d, array_3d)

        # Prediction
        # area_list = Pred.Calculate_2D_Area(boundary_point_list, th_index)
        weight_list = Pred.Calculate_Weight(area_list)

        # Visualization
        Visual.Build_PNG(filename, start_date, results, area_list,
                        weight_list, th_index, cfg_file, check_file)
        Visual.Save_CSV(filename, start_date, area_list, weight_list)

if __name__ == "__main__":
    main()