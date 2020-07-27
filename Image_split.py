# coding=utf-8
# 图片切割按照2*3进行切割
import os
from PIL import Image


# 切割图片的函数
def splitimage(src, rownum, colnum, dstpath):
    img = Image.open(src)
    w, h = img.size
    if rownum <= h and colnum <= w:
        print('Original image info: %sx%s, %s, %s' % (w, h, img.format, img.mode))
        print('开始处理图片切割, 请稍候...')
        # 按照路径将文件名和路径分隔开
        s = os.path.split(src)
        if dstpath == '':
            dstpath = s[0]
        fn = s[1].split('.')
        basename = fn[0]
        ext = fn[-1]

        num = 0
        rowheight = h // rownum
        colwidth = w // colnum
        for r in range(rownum):
            for c in range(colnum):
                box = (c * colwidth, r * rowheight, (c + 1) * colwidth, (r + 1) * rowheight)
                img.crop(box).save(os.path.join(dstpath, basename + '_' + str(num) + '.' + ext), ext)
                num = num + 1

        print('图片切割完毕，共生成 %s 张小图片。' % num)
    else:
        print('不合法的行列切割参数！')


# 存放图片的文件夹
folder = 'C:\\Users\\User\\Desktop\\rice\\test'
# os.listdir()读取文件夹中的文件
path = os.listdir(folder)
#print(path)
# 定义要创建的目录
mkpath = 'C:\\Users\\User\\Desktop\\rice\\test\\before'

dstpath = mkpath
# each_bmp为每张图片的路径
for each_bmp in path:
    first_name, second_name = os.path.splitext(each_bmp)
    # 切割图片命名
    # os.path.join()函数用于路径拼接文件路径。
    # os.path.join()函数中可以传入多个路径：
    each_bmp = os.path.join(folder,each_bmp)
    src = each_bmp
    # src为每张图片的路径
    print(src)
    # first_name为图片名称
    print(first_name)
    # 遍历，来进行批量操作
    if os.path.isfile(src):
        # 切割行数
        row = int(2)
        # 切割列数
        col = int(3)
        if row > 0 and col > 0:
            splitimage(src, row, col, dstpath)
        else:
            print('无效的行列切割参数！')
    else:
        print('图片文件 %s 不存在！' % src)