from Unetpp_scSE_2 import Unetpp_scSE
from unetpp import NestedUNet
from unet import UNet
from dataset import *
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from modules import *
from save_history import *



if __name__ == "__main__":
    root_png = "F:\\XiaohanYuan_Data\\test\\chnxiaoqing"
    # for time in ["010"]:
    for time in ["010","020","030","040","050","060","070","080","090","100"]:
        test = DataVal(
            os.path.join(root_png, time, "BG", "A"),
            os.path.join(root_png, time, "masks"))  # val data


        # Dataloader begins
        test_load = \
            torch.utils.data.DataLoader(dataset=test,
                                        num_workers=0, batch_size=1, shuffle=False)
        # Dataloader end

        # Model
        # model = UNet(in_channels=1, out_channels=5)
        # model = Unetpp_scSE(in_channels=1, out_channels=5)
        model = NestedUNet(in_channels=1, out_channels=5)
        
        model = torch.nn.DataParallel(model, device_ids=list(
            range(torch.cuda.device_count()))).cuda()

        model.load_state_dict(torch.load("D:\\XiaohanYuan\\multiclass_unetpp_deepsup\Histories\\01\\saved_models\\model_epoch_60.pkl"))


        # Saving History
        if not os.path.exists("./history"):
            os.makedirs("./history")
        image_save_path = "./history/"+time


        # Saving History to csv
        header = ['time', 
                  'DSC_LA', 'DSC_LV', 'DSC_RA', 'DSC_RV', 
                  'Iou_LA', 'Iou_LV', 'Iou_RA', 'Iou_RV', 
                  'PA_LA', 'PA_LV', 'PA_RA', 'PA_RV', ]
        save_file_name = "./history/history.csv"
        save_dir = "./history/"


        test_dice, test_acc, test_iou = test_model(model, test_load, image_save_path)  # 对test预测
        print(time)
        print("test dice:", test_dice)
        print("test iou:", test_iou)
        print("test acc:", test_acc)
        
        values = [time]
        values = np.append(np.append(np.append(values, test_dice), test_iou), test_acc)
        export_history(header, values, save_dir, save_file_name)
