import os
import shutil

path = 'D:\EdoublemA\rice\Emma-Semantic-Segmentation\chalkynet - test\dataset2'
new_path = 'D:\\EdoublemA\\rice\\Emma-Semantic-Segmentation\\Unet_Mobile - 0314\\dataset2\\png'
count = os.listdir(path)
for j in range(1, len(count) + 1):
    for root, dirs, files in os.walk(path):
        if len(dirs) == 0:
            for i in range(len(files)):
                print("i=", i)
                if files[i].find('label.png') != -1:
                    shutil.copy(os.path.join(path + str(j).zfill(1) + '_json', files[i]),
                                os.path.join(new_path, str(j).zfill(1) + '.png'))

# path为包含所有要提取的png文件的原文件夹名
