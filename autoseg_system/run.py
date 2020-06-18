import torch
import torch.nn as nn
import numpy as np
import nrrd
from unet import UNet
from util import *
import time

def Process(filename, foldername):
    start = time.perf_counter()
    # 输入背景nrrd路径
    # root_nrrd = "D:\\XiaohanYuan\\autoseg-system\\model\\BG.nrrd"
    root_nrrd = filename
    # 输出分割后nrrd路径
    # out_nrrd = "D:\\XiaohanYuan\\autoseg-system\\model"
    out_nrrd = foldername

    # Pre-processing:nrrd2mat
    readdata_bg, header_bg = nrrd.read(root_nrrd)
    map_bg = np.zeros(readdata_bg.shape)
    for index in range(readdata_bg.shape[2]):
        bg = readdata_bg[:, :, index]
        bg = ImgTrasform(bg)
        bg = Normalization(windowAdjust(bg, 800, 200)) * 255
        map_bg[:, :, index] = bg


    data = Data(map_bg)
    data_load = torch.utils.data.DataLoader(dataset=data, num_workers=0, batch_size=1, shuffle=False)
    model = UNet(in_channels=1, out_channels=5)
    model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()

    model.load_state_dict(torch.load("D:\\XiaohanYuan\\autoseg-system\\model\\model_epoch_100.pkl"))
    pred = test_model(model, data_load)

    size = (pred.shape[0], pred.shape[1])
    # 分别保存为四个腔室nrrd文件
    la = np.zeros(pred.shape).astype(np.uint8)
    for index in range(pred.shape[2]):
        img = np.zeros(size).astype(np.uint8)
        list = np.where(pred[:,:,index].astype(np.int32) == 1)
        img[list] = 255
        # 重新镜像旋转变换回去再转化为nrrd
        img = Inverse_ImgTrasform(img)
        la[:, :, index] = img
    nrrd.write(out_nrrd + '\\LA-label.nrrd', la, header=header_bg)


    lv = np.zeros(pred.shape).astype(np.uint8)
    for index in range(pred.shape[2]):
        img = np.zeros(size).astype(np.uint8)
        list = np.where(pred[:, :, index].astype(np.int32) == 2)
        img[list] = 255
        img = Inverse_ImgTrasform(img)
        lv[:, :, index] = img
    nrrd.write(out_nrrd + '\\LV-label.nrrd', lv, header=header_bg)


    ra = np.zeros(pred.shape).astype(np.uint8)
    for index in range(pred.shape[2]):
        img = np.zeros(size).astype(np.uint8)
        list = np.where(pred[:, :, index].astype(np.int32) == 3)
        img[list] = 255
        img = Inverse_ImgTrasform(img)
        ra[:, :, index] = img
    nrrd.write(out_nrrd + '\\RA-label.nrrd', ra, header=header_bg)


    rv = np.zeros(pred.shape).astype(np.uint8)
    for index in range(pred.shape[2]):
        img = np.zeros(size).astype(np.uint8)
        list = np.where(pred[:, :, index].astype(np.int32) == 4)
        img[list] = 255
        img = Inverse_ImgTrasform(img)
        rv[:, :, index] = img
    nrrd.write(out_nrrd + '\\RV-label.nrrd', rv, header=header_bg)

    end = time.perf_counter()

    return str(round(end-start))

if __name__=="__main__":
    Process("D:\\XiaohanYuan\\autoseg-system\\model\\BG.nrrd", "D:\\XiaohanYuan\\autoseg-system\\model")
