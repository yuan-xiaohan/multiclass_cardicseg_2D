import numpy as np
import cv2
import random
import torch
from Unetpp_scSE_2 import Unetpp_scSE
import matplotlib
import matplotlib.pyplot as plt
import os

images_root = "E:\\yuanxiaohan\\Cardic_segmentation\\my project\\test\\draw\\compare\\images"
masks_root = "E:\\yuanxiaohan\\Cardic_segmentation\\my project\\test\\draw\\compare\\masks"
blendings_root = "E:\\yuanxiaohan\\Cardic_segmentation\\my project\\test\\draw\\compare\\blendings"

images_file = os.listdir(images_root)
masks_file = os.listdir(masks_root)
for index in range(len(images_file)):
    img = cv2.imread(os.path.join(images_root, images_file[index]),)
    mask = cv2.imread(os.path.join(masks_root, masks_file[index]), 0)

    mask_color = np.zeros([512, 512, 3]).astype(np.uint8)

    # LA
    Slice = np.zeros([512, 512]).astype(np.uint8)
    mask_new = np.zeros([512, 512, 3]).astype(np.uint8)
    list = np.where(mask.astype(np.int32) == 50)
    Slice[list] = 145
    mask_new[:, :, 0] = Slice
    Slice[list] = 214
    mask_new[:, :, 1] = Slice
    Slice[list] = 241
    mask_new[:, :, 2] = Slice
    mask_color = mask_color + mask_new

    # LV
    Slice = np.zeros([512, 512]).astype(np.uint8)
    mask_new = np.zeros([512, 512, 3]).astype(np.uint8)
    list = np.where(mask.astype(np.int32) == 100)
    Slice[list] = 84
    mask_new[:, :, 0] = Slice
    Slice[list] = 110
    mask_new[:, :, 1] = Slice
    Slice[list] = 206
    mask_new[:, :, 2] = Slice
    mask_color = mask_color + mask_new

    # RA
    Slice = np.zeros([512, 512]).astype(np.uint8)
    mask_new = np.zeros([512, 512, 3]).astype(np.uint8)
    list = np.where(mask.astype(np.int32) == 150)
    Slice[list] = 210
    mask_new[:, :, 0] = Slice
    Slice[list] = 184
    mask_new[:, :, 1] = Slice
    Slice[list] = 111
    mask_new[:, :, 2] = Slice
    mask_color = mask_color + mask_new

    # RV
    Slice = np.zeros([512, 512]).astype(np.uint8)
    mask_new = np.zeros([512, 512, 3]).astype(np.uint8)
    list = np.where(mask.astype(np.int32) == 200)
    Slice[list] = 128
    mask_new[:, :, 0] = Slice
    Slice[list] = 174
    mask_new[:, :, 1] = Slice
    Slice[list] = 128
    mask_new[:, :, 2] = Slice
    mask_color = mask_color + mask_new

    # cv2.imshow('color', mask_color)


    blending = cv2.addWeighted(img, 0.6, mask_color, 0.4, 0)
    cv2.imwrite(blendings_root + "\\" +  str(index) + ".png",blending)
    # cv2.imshow('Blending', blending)

    # cv2.waitKey(0)