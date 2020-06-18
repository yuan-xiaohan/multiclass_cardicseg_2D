import numpy as np
import cv2
import torch
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset


# 背景顺时针90度旋转 + 水平镜像
def ImgTrasform(img):
    row, col = img.shape[:2]
    M = cv2.getRotationMatrix2D((col / 2, row / 2), -90, 1)
    img_new = cv2.flip(cv2.warpAffine(img, M, (col, row)), 1)
    return img_new

# 反变换：水平镜像 + 逆时针90度旋转
def Inverse_ImgTrasform(img):
    row, col = img.shape[:2]
    img = cv2.flip(img, 1)
    M = cv2.getRotationMatrix2D((col / 2, row / 2), 90, 1)
    img_new =cv2.warpAffine(img, M, (col, row))
    return img_new

# 映射到0~1之间
def Normalization(hu_value):
    hu_min = np.min(hu_value)
    hu_max = np.max(hu_value)
    normal_value = (hu_value - hu_min) / (hu_max - hu_min)
    return normal_value

# 根据窗宽、窗位计算出窗的最大值和最小值
def windowAdjust(img,ww,wl):
    win_min = wl - ww / 2
    win_max = wl + ww / 2
    # 根据窗最大值、最小值来截取img
    img_new = np.clip(img,win_min,win_max)
    return img_new


class Data(Dataset):
    def __init__(self, data):
        self.data = data
        self.data_len = self.data.shape[2]

    def __getitem__(self, index):
        img = self.data[:, :, index]
        img = np.expand_dims(img, axis=0)  # add additional dimension
        img_as_tensor = torch.from_numpy(img).float()  # Convert numpy array to tensor

        return img_as_tensor

    def __len__(self):
        return self.data_len


def test_model(model, data_load):
    stacked_img = []
    for batch, (images) in enumerate(data_load):
        with torch.no_grad():
            img = Variable(images.cuda())  # (bs,1,512,512) 0,1,2,3,4
            output = model(img)  # (bs,5,512,512)

        #pred = torch.argmax(output[3], dim=1) #deep_supervision
        pred = torch.argmax(output, dim=1) #(bs,512,512) 0,1,2,3,
        pred_np = pred[0, :, :].cpu().numpy()
        stacked_img.append(pred_np)
    stacked_img = np.array(stacked_img)
    stacked_img = stacked_img.transpose((1, 2, 0))
    return stacked_img
