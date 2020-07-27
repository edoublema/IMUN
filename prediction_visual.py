from nets.unet import mobilenet_unet
#from nets.uunet import get_unet
from PIL import Image
import numpy as np
import random
import copy
import os
import time
import cv2

from timeit import default_timer as timer

tic = timer()

# 待测试的代码
# 待测试的代码
# 待测试的代码

#二分类
NCLASSES = 4
HEIGHT = 256
WIDTH = 256

colors = [[0, 0, 0], [0, 255, 0],[128, 128, 0],[128, 0, 0]]
#model = get_unet(4,256,256)
model = mobilenet_unet(n_classes=NCLASSES,input_height=HEIGHT, input_width=WIDTH)

model.load_weights("logs/ep033-loss1.488-val_loss1.936.h5")
imgs = os.listdir("./img")

for jpg in imgs:

    img = Image.open("./img" + "\\" + jpg)
    old_img = copy.deepcopy(img)
    orininal_h = np.array(img).shape[0]
    orininal_w = np.array(img).shape[1]

    img = img.resize((WIDTH,HEIGHT))
    img = np.array(img)
    img = img/255
    img = img.reshape(-1,HEIGHT,WIDTH,3)
    pr = model.predict(img)[0]

    pr = pr.reshape((int(HEIGHT), int(WIDTH),NCLASSES)).argmax(axis=-1)

    seg_img = np.zeros((int(HEIGHT), int(WIDTH),3))

    for c in range(NCLASSES):
        try:
            seg_img[:,:,0] += ( (pr[:,: ] == c )*( colors[c][0] )).astype('uint8')
            seg_img[:,:,1] += ((pr[:,: ] == c )*( colors[c][1] )).astype('uint8')
            seg_img[:,:,2] += ((pr[:,: ] == c )*( colors[c][2] )).astype('uint8')
        except:
            continue

    #seg_img = cv2.resize(seg_img, (256,256))
    cv2.imwrite("./img_out" + "\\" + jpg, seg_img)

toc = timer()

print(toc - tic) # 输出的时间，秒为单位
