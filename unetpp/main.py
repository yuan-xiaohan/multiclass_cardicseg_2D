#from model_unet1 import U_Net #unet1
from unetpp import NestedUNet #unet2
from dataset import *
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from PIL import Image 
import cv2
from modules import *
from save_history import *
from losses import LovaszLossSoftmax
from losses import LovaszLossHinge


if __name__ == "__main__":

    # Dataset begin
    train = DataTrain(
        'F:/XiaohanYuan_Data/muticlass_pick/train/images',
        'F:/XiaohanYuan_Data/muticlass_pick/train/masks')  # train data
    val = DataVal(
        'F:/XiaohanYuan_Data/muticlass_pick/val/images',
        'F:/XiaohanYuan_Data/muticlass_pick/val/masks')  # val data

    # Dataloader begins
    train_load = \
        torch.utils.data.DataLoader(dataset=train,
                                    num_workers=0, batch_size=4, shuffle=True)
    val_load = \
        torch.utils.data.DataLoader(dataset=val,
                                    num_workers=0, batch_size=1, shuffle=False)
    # Dataloader end

# Model
    model = NestedUNet(in_channels=1, out_channels=5)
    # model = U_Net()
    model = torch.nn.DataParallel(model, device_ids=list(
        range(torch.cuda.device_count()))).cuda()

    # model.load_state_dict(torch.load("E:\\yuanxiaohan\\Cardic_segmentation\\my project\\LV_seg_all\\Histories\\02\\saved_models\\model_epoch_checkpoint_60.pth"))

    # Loss function
    criterion = LovaszLossSoftmax()

    # Optimizerd
    # optimizer = torch.optim.RMSprop(model.module.parameters(), lr=1e-4)
    optimizer = torch.optim.Adam(model.module.parameters(),
                                 lr=1e-4,
                                 betas=(0.9, 0.999),
                                 eps=1e-8,
                                 weight_decay=0)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[20, 50],
                                                     gamma=0.9)

    # Parameters
    epoch_start = 0
    epoch_end = 1000

    # Saving History to csv
    header = ['epoch', 'train loss', 'train dice_LA', 'train dice_LV', 'train dice_RA', 'train dice_RV', 'val loss',
              'val dice_LA', 'val dice_LV', 'val dice_RA', 'val dice_RV']
    save_file_name = "./history/history.csv"
    save_dir = "./history/"

    # Saving images and models directories
    model_save_dir = "./history/saved_models"
    image_save_path = "./history/result_images"

    # Train
    print("Initializing Training!")
    for i in range(epoch_start, epoch_end):
        # train the model
        train_model(model, train_load, criterion, optimizer, scheduler)  # 正式训练
        train_dice, train_loss = get_loss_train(model, train_load, criterion)  # 计算训练的loss

        # train_loss = train_loss / len(train)
        print('Epoch', str(i+1), 'Train loss:', train_loss, "\n", "Train dice", train_dice)  # 输出每一代的loss和acc

        # Validation every 5 epoch 每5代输出一次val集的loss和dice
        if (i+1) % 5 == 0:
            val_dice, val_loss = validate_model(
                model, val_load, criterion, i+1, True, image_save_path)  # 对val预测
            print('Val loss:', val_loss, "val dice:", val_dice)
            values = [i + 1, train_loss]
            values = np.append(np.append(np.append(values, train_dice), val_loss), val_dice)
            export_history(header, values, save_dir, save_file_name)  # 将每5代的结果保存
            
            if (i+1) % 20 == 0:  # save model every 10 epoch 每100代输出一次模型
                save_models(model, model_save_dir, i+1, optimizer)
