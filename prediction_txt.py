from nets.unet import mobilenet_unet
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

colors = [[0, 0, 0], [0, 255, 0],[255, 0, 0],[0, 0, 255]]
model = mobilenet_unet(n_classes=NCLASSES,input_height=HEIGHT, input_width=WIDTH)

model.load_weights("logs/ep030-loss0.439-val_loss0.774.h5")
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

    #seg_img = np.zeros((int(HEIGHT), int(WIDTH),3))
    seg_img = pr

    for c in range(NCLASSES):
        try:
            seg_img[:,:,0] += ( (pr[:,: ] == c )*( colors[c][0] )).astype('uint8')
            seg_img[:,:,1] += ((pr[:,: ] == c )*( colors[c][1] )).astype('uint8')
            seg_img[:,:,2] += ((pr[:,: ] == c )*( colors[c][2] )).astype('uint8')
        except:
            continue

    #cv2.imwrite("./output/" + "\\" + jpg, seg_img)
    print(jpg, seg_img.max())
    #print(seg_img.shape)
    np.savetxt('./output/{}.txt'.format(jpg[:-4]), seg_img,fmt='%0.f')
    print(type(seg_img))
    #seg_img = Image.fromarray(np.uint8(seg_img))
    #img.save("./img_out/" + "\\" + jpg, seg_img)
    #print(jpg, seg_img.max())
    #print(type(seg_img))
    #print(jpg, seg_img.max())
    # plt.figure("1.jpg")
    # plt.imshow(seg_img)
    # plt.axis('off')
    # plt.show()

    # # print(img)
    # plt.figure("1.jpg")
    # plt.imshow(img)
    #
    # plt.axis('off')
    # plt.show()
    # exit()
    # img = Image.fromarray(seg_img)
    # img.save(imgs + "\\" + jpg)
toc = timer()

print(toc - tic) # 输出的时间，秒为单位
