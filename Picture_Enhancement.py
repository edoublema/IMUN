# coding=utf-8
import os
from PIL import Image
from skimage import transform
from numpy import *
import numpy as np
import cv2


class BatchRename():
    '''
    批量重命名文件夹中的图片文件
    '''
    def __init__(self):
        self.path = "C:\\Users\\User\\Desktop\\testrice"

    def rename(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)
        i = 1
        for item in filelist:
            if item.endswith('.bmp'):
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path), str(i) + '.jpg')
                try:
                    os.rename(src, dst)
                    print('converting %s to %s ...' % (src, dst))
                    i += 1
                except:
                    continue
        print('total %d to rename & converted %d jpgs' % (total_num, i))


# bmp 转换为jpg
# 左右镜像
# 上下镜像
# 旋转180度
def Enhancement(file_path):
   for fileName in os.listdir(file_path):
       print(fileName)
       im = Image.open(file_path + "\\"+fileName)

       # 左右镜像
       mirror_lr = im.transpose(Image.FLIP_LEFT_RIGHT)
       mirror_lr.save(file_path + "\\" + 'mirror_lr' + fileName)

       # 上下镜像
       mirror_tb = im.transpose(Image.FLIP_TOP_BOTTOM)
       mirror_tb.save(file_path + "\\" + 'mirror_tb' + fileName)

       # 旋转180度
       rotate = im.transpose(Image.ROTATE_180)
       rotate.save(file_path + "\\" + 'rotate' + fileName)


# 注意：使用cv2.imread时文件夹内若有其他格式文件会报错
def resize(file_path):
    for fileName in os.listdir(file_path):
       img = cv2.imread(file_path+"\\"+fileName)
       dst = cv2.resize(img, (256, 192), interpolation=cv2.INTER_CUBIC)
       cv2.imwrite(file_path + "\\" + fileName, dst)


def cropped(file_path):
    for fileName in os.listdir(file_path):
        img = cv2.imread(file_path + "\\" + fileName)
        # 裁剪坐标为[y0:y1, x0:x1]
        cropped = img[32:256, 32:256]
        cv2.imwrite(file_path + "\\" + fileName, cropped)


def gray(file_path):
    for fileName in os.listdir(file_path):
        img = Image.open(file_path + "\\" + fileName)
        rgb = img.convert("L")
        rgb.save(file_path + "\\" + 'gray' + fileName)
        gray = Image.open(file_path + "\\" + 'gray' + fileName).convert('RGB')
        gray.save(file_path + "\\" + fileName)



def color(file_path):
    for fileName in os.listdir(file_path):
        img = Image.open(file_path + "\\" + fileName)
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        hsv.save(file_path + "\\" + 'hsv' + fileName)
        #gray = Image.open(file_path + "\\" + 'gray' + fileName).convert('RGB')
        #gray.save(file_path + "\\" + 'gray' + fileName)





# 删除原来的位图
def deleteImages(file_path, file_list):
    """
    删除图片
    """
    for fileName in file_list:
        command = "del " + file_path + "\\*." + 'bmp'
        os.system(command)


def main():
   file_path = './GT1'
   resize(file_path)
   #Enhancement(file_path)
   #gray(file_path)
   #color(file_path)
   #file_list = os.listdir(file_path)
   # deleteImages(file_path, file_list)
   #cropped(file_path)


if __name__ == '__main__':
    main()
    # demo = BatchRename()
    # demo.rename()