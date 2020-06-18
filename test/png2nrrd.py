import cv2
import numpy as np
import copy
import os
import nrrd

# 反变换：水平镜像 + 逆时针90度旋转
def Inverse_ImgTrasform(img):
    row, col = img.shape[:2]
    img = cv2.flip(img, 1)
    M = cv2.getRotationMatrix2D((col / 2, row / 2), 90, 1)
    img_new =cv2.warpAffine(img, M, (col, row))
    return img_new

# pick png to 3D mat
def png2mat(out_dir, sorted_file):
        num = len(sorted_file)
        matrix = np.zeros((512, 512, num), dtype=np.uint8)
        n = 0
        for in_name in sorted_file:
            in_content_path = os.path.join(out_dir, in_name)
            matrix[:, :, n] = cv2.imread(in_content_path)[:, :, 1]
            n = n + 1
        return matrix

in_bg_dir = "E:\\yuanxiaohan\\Cardic_segmentation\\data\\WHS-ten-nrrd\\chnxiaoqing\\070\\BG.nrrd"
readdata_bg, header_bg = nrrd.read(in_bg_dir)
print(readdata_bg.shape)
print(header_bg)

mask_dir = "E:\\yuanxiaohan\\Cardic_segmentation\\my project\\test\\unet\\100epoch\\070"
# data = np.zeros((512, 512, 192)).astype(np.short)
mask = png2mat(mask_dir, os.listdir(mask_dir))

la = np.zeros((512, 512, mask.shape[2])).astype(np.uint8)
cv2.imshow('LA_1',la[:,:,80])
for index in range(mask.shape[2]):
    img = np.zeros((512, 512)).astype(np.uint8)
    list = np.where(mask[:,:,index].astype(np.int32) == 50)
    img[list] = 255
    # 重新镜像旋转变换回去再转化为nrrd
    img = Inverse_ImgTrasform(img) 
    la[:,:,index]=img
nrrd.write('LA-label.nrrd', la, header=header_bg)


lv = np.zeros((512, 512, mask.shape[2])).astype(np.uint8)
for index in range(mask.shape[2]):
    img = np.zeros((512, 512)).astype(np.uint8)
    list = np.where(mask[:,:,index].astype(np.int32) == 100)
    img[list] = 255
    img = Inverse_ImgTrasform(img) 
    lv[:,:,index]=img
nrrd.write('LV-label.nrrd', lv, header=header_bg)


ra = np.zeros((512, 512, mask.shape[2])).astype(np.uint8)
for index in range(mask.shape[2]):
    img = np.zeros((512, 512)).astype(np.uint8)
    list = np.where(mask[:,:,index].astype(np.int32) == 150)
    img[list] = 255
    img = Inverse_ImgTrasform(img) 
    ra[:,:,index]=img
nrrd.write('RA-label.nrrd', ra, header=header_bg)


rv = np.zeros((512, 512, mask.shape[2])).astype(np.uint8)
for index in range(mask.shape[2]):
    img = np.zeros((512, 512)).astype(np.uint8)
    list = np.where(mask[:,:,index].astype(np.int32) == 200)
    img[list] = 255
    img = Inverse_ImgTrasform(img) 
    rv[:,:,index]=img
nrrd.write('RV-label.nrrd', rv, header=header_bg)


'''
# Read the data back from file
readdata, header = nrrd.read('testdata.nrrd')
print(readdata.shape)
print(header)
'''