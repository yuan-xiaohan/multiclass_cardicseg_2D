import numpy as np
from PIL import Image
import torch

# dice coefficient
def dice_check(mask, prediction):

    smooth = 1.
    num = mask.size(0)
    m1 = mask.view(num, -1).float()  # Flatten
    m2 = prediction.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()
    dice = (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

    # 输出为tensor
    return dice.item()

def dice_check_for_batch(masks, predictions, batch_size):
    total_dice = 0
    for index in range(batch_size):
        total_dice += dice_check(masks[index], predictions[index])
    return total_dice/batch_size



# accuracy
def accuracy_check(mask, prediction):
    ims = [mask, prediction]
    np_ims = []
    for item in ims:
        if 'str' in str(type(item)):
            item = np.array(Image.open(item))
        elif 'PIL' in str(type(item)):
            item = np.array(item)
        elif 'torch' in str(type(item)):
            item = item.numpy()
        np_ims.append(item)

    compare = np.equal(np_ims[0], np_ims[1])
    accuracy = np.sum(compare)

    return accuracy/len(np_ims[0].flatten())


def accuracy_check_for_batch(masks, predictions, batch_size):
    total_acc = 0
    for index in range(batch_size):
        total_acc += accuracy_check(masks[index], predictions[index])
    return total_acc/batch_size


# iou
def iou_check(mask, prediction):

    smooth = 1.
    num = mask.size(0)
    m1 = mask.view(num, -1).float()  # Flatten
    m2 = prediction.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()
    union = m1.sum() + m2.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)

    # 输出为tensor
    return iou.item()

def iou_check_for_batch(masks, predictions, batch_size):
    total_iou = 0
    for index in range(batch_size):
        total_iou += iou_check(masks[index], predictions[index])
    return total_iou/batch_size






