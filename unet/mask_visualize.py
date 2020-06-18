import cv2
import numpy as np
import os


dir_input = "F:\\XiaohanYuan_Data\\muticlass_mini\\val\\masks\\"
dir_output = "F:\XiaohanYuan_Data\muticlass_mini\\val\masks_visualize\\"
for index_dir in os.listdir(dir_input):
    print(index_dir)
    mask = cv2.imread(dir_input + index_dir)
    mask = mask * 50
    cv2.imwrite(dir_output + index_dir, mask)