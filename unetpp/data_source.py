import cv2
import numpy as np
import os
import csv
import shutil

# pick png to 3D mat
def png2mat_pick(out_dir, sorted_file):
    num = len(sorted_file)
    matrix = cv2.imread(os.path.join(out_dir, sorted_file[0]))[:, :, 0][:, :, np.newaxis]
    for index in range(2, num, 2):
        in_content_path = os.path.join(out_dir, sorted_file[index])
        matrix = np.append(matrix, cv2.imread(in_content_path)[:, :, 0][:, :, np.newaxis], axis=2)
    return matrix

def changenum(i):
    if i < 10:
        j = '000' + str(i)
    elif (i > 9 and i < 100):
        j = '00' + str(i)
    else:
        j = '0' + str(i)
    return j



root_png = "E:\\yuanxiaohan\\Cardic_segmentation\\data\\WHS-ten-png"
bg_out_dir = "E:\\yuanxiaohan\\Cardic_segmentation\\my project\\muliclass_unetpp\\data\\val\\images"
mask_out_dir = "E:\\yuanxiaohan\\Cardic_segmentation\\my project\\muliclass_unetpp\\data\\val\\masks"


#for pair in [["huxiaoying", "02"], ["jiangzhongyin", "03"], ["lanjunfang", "04"],
#              ["liyanping", "05"], ["shadebin", "06"], ["wuyong", "07"], ["xialianglluan", "08"], ["zhuyuyin", "09"]]:
for pair in [["chnxiaoqing", "01"]]:

    pname = pair[0]
    pnumber = pair[1]
    print(pname, pnumber)


    # for time in ["010", "020", "030", "040", "050", "060", "070", "080", "090"]:
    for time in ["100"]:
        print(time)
        bg_dir = os.path.join(root_png, pname, time, "BG", "A")
        mask_la_dir = os.path.join(root_png, pname, time, "LA", "A")
        mask_lv_dir = os.path.join(root_png, pname, time, "LV", "A")
        mask_ra_dir = os.path.join(root_png, pname, time, "RA", "A")
        mask_rv_dir = os.path.join(root_png, pname, time, "RV", "A")


        bg = png2mat_pick(bg_dir, os.listdir(bg_dir))
        mask_la = png2mat_pick(mask_la_dir, os.listdir(mask_la_dir))
        mask_lv = png2mat_pick(mask_lv_dir, os.listdir(mask_lv_dir))
        mask_ra = png2mat_pick(mask_ra_dir, os.listdir(mask_ra_dir))
        mask_rv = png2mat_pick(mask_rv_dir, os.listdir(mask_rv_dir))

        mask_la = mask_la / 255 * 1
        mask_lv = mask_lv / 255 * 2
        mask_ra = mask_ra / 255 * 3
        mask_rv = mask_rv / 255 * 4
        mask = mask_la + mask_lv + mask_ra + mask_rv

        for index in range(bg.shape[2]):
            img = bg[:, :, index]
            cv2.imwrite(bg_out_dir + "\\" + pnumber + "_" + time + "_" + changenum(index + 1) + ".png", img)

        for index in range(mask.shape[2]):
            img = mask[:, :, index]
            cv2.imwrite(mask_out_dir + "\\" + pnumber + "_" + time + "_" + changenum(index + 1) + ".png", img)

        # list_bg = os.listdir(bg_dir)
        # for index in range(0, bg.shape[-1], 2):
        #     shutil.copyfile(os.path.join(bg_dir, list_bg[index]), os.path.join(root_out, "BG", list_bg[index]))
