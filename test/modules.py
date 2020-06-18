from dataset import *
import torch.nn as nn
from metrics import *
import os
from torch.autograd import Variable


# 测试test
def test_model(model, data_val, save_folder_name):
    """
        Validation run
    """
    # calculating validation loss
    total_test_dice = np.zeros(4)
    total_test_acc = np.zeros(4)
    total_test_iou = np.zeros(4)
    for batch, (images_v, masks_v) in enumerate(data_val):
        with torch.no_grad():
            images_v = Variable(images_v.cuda())
            masks_v = Variable(masks_v.cuda()) # (bs,1,512,512) 0,1,2,3,4
            outputs_v = model(images_v) # (bs,5,512,512)

        # save to image or not
        preds_v = torch.argmax(outputs_v[3], dim=1) #deep_supervision
        #preds_v = torch.argmax(outputs_v, dim=1) #(bs,512,512) 0,1,2,3,4
        save_prediction_image(preds_v, batch, save_folder_name)

        outputs_final = torch.softmax(outputs_v[3], dim=1) #deep_supervision
        # outputs_final = torch.softmax(outputs_v, dim=1)
        dice_list = []
        acc_list = []
        iou_list = []

        for class_index in range(1, outputs_final.shape[1]): 
            mask = np.zeros(masks_v.shape)
            mask[np.where(masks_v.cpu() == class_index)] = 1
            mask = torch.from_numpy(mask) # (bs,1,512,512) 0,1

            preds_v_new = torch.unsqueeze(preds_v, 1)# (bs,1,512,512)
            preds_acc = np.zeros(masks_v.shape)
            preds_acc[np.where(preds_v_new.cpu() == class_index)] = 1
            preds_acc = torch.from_numpy(preds_acc) # (bs,1,512,512) 0,1

            # dice
            dicecoeff = dice_check_for_batch(mask.cpu().float(), outputs_final[:, class_index, :, :].cpu(),
                                                 images_v.size()[0])
            dice_list = np.append(dice_list, dicecoeff)
            # accuracy
            acc = accuracy_check_for_batch(mask.cpu().float(), preds_acc.cpu(),
                                                 images_v.size()[0])
            acc_list = np.append(acc_list, acc)
            #iou
            iou = iou_check_for_batch(mask.cpu().float(), outputs_final[:, class_index, :, :].cpu(),
                                                 images_v.size()[0])
            iou_list = np.append(iou_list, iou)

        total_test_dice = total_test_dice + dice_list
        total_test_acc = total_test_acc + acc_list
        total_test_iou = total_test_iou + iou_list

    return total_test_dice / (batch + 1), total_test_acc / (batch + 1), total_test_iou / (batch + 1)


def save_prediction_image(preds_v, im_name, save_folder_name):
    """save images to save_path
    """
    img = preds_v.cpu().data.numpy()  # (1,256,256)
    # print(type(img),img.shape)
    img = img.transpose((1, 2, 0))  # (256,256,1)
    img = np.squeeze(img, axis=2)  # 保留前两维(256,256)
    img = img * 50
    img_np = img.astype('uint8')
    img = Image.fromarray(img_np, mode='L')  # 生成灰度图
    # img.show()

    desired_path = save_folder_name
    # Create the path if it does not exist
    if not os.path.exists(desired_path):
        os.makedirs(desired_path)
    # Save Image!
    export_name = changenum(im_name+1) + '.png'
    img.save(desired_path +'/'+ export_name)


def polarize(img):
    ''' Polarize the value to zero and one
    Args:
        img (numpy): numpy array of image to be polarized
    return:
        img (numpy): numpy array only with zero and one
    '''
    img[img >= 0.5] = 1
    img[img < 0.5] = 0
    return img

def changenum(i):
    if i < 10:
        j = '00' + str(i)
    elif (i > 9 and i < 100):
        j = '0' + str(i)
    else:
        j = str(i)
    return j