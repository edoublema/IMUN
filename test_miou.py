import numpy as np
import argparse
import json
from PIL import Image
from os.path import join
__all__ = ['SegmentationMetric']

"""
confusionMetric  # 注意：此处横着代表预测值，竖着代表真实值，与之前介绍的相反
P\L     P    N
P      TP    FP
N      FN    TN
"""


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89


    def genConfusionMatrix(self, imgPredict, imgLabel):  # 同FCN中score.py的fast_hist()函数
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (
                np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) -
                np.diag(self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))


if __name__ == '__main__':
    gt_dir = './GT/'
    pred_dir =  './output'
    devkit_dir = './txt/'
    image_path_list = join(devkit_dir, 'val2.txt')  # 在这里打开记录分割图片名称的txt
    label_path_list = join(devkit_dir, 'val2.txt')  # ground truth和自己的分割结果txt一样
    gt_imgs = open(label_path_list, 'r').read().splitlines()  # 获得验证集标签名称列表
    gt_imgs = [join(gt_dir, x) for x in gt_imgs]  # 获得验证集标签路径列表，方便直接读取
    pred_imgs = open(image_path_list, 'r').read().splitlines()  # 获得验证集图像分割结果名称列表
    pred_imgs = [join(pred_dir, x[:-3] + 'txt') for x in pred_imgs]
    for ind in range(len(gt_imgs)):  # 读取每一个（图片-标签）对
        # pred = np.array(Image.open(pred_imgs[ind]))  # 读取一张图像分割结果，转化成numpy数组
        label = np.array(Image.open(gt_imgs[ind]))  # 读取一张对应的标签，转化成numpy数组
        # print pred.shape
        # print label.shape
        pred = np.loadtxt(pred_imgs[ind]).astype(np.uint8)
        # label = label_mapping(label, mapping)#进行标签映射（因为没有用到全部类别，因此舍弃某些类别），可忽略
        if len(label.flatten()) != 0 and len(pred.flatten()) != 0:
            if len(label.flatten()) != len(pred.flatten()):  # 如果图像分割结果与标签的大小不一样，这张图片就不计算

                print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()),
                                                                                  len(pred.flatten()), gt_imgs[ind],
                                                                                  pred_imgs[ind]))
                continue
    num_classes = 4
    print('Num classes', num_classes)
    imgPredict = np.loadtxt(pred_imgs[ind]).astype(np.uint8)  # 可直接换成预测图片
    imgLabel =np.array(Image.open(gt_imgs[ind]))  # 可直接换成标注图片
    metric = SegmentationMetric(4)  # 4表示有4个分类，有几个分类就填几
    metric.addBatch(imgPredict, imgLabel)
    pa = metric.pixelAccuracy()
    #cpa = metric.classPixelAccuracy()
    mpa = metric.meanPixelAccuracy()
    print('pa is : %f' % pa)
    #print('cpa is :')  # 列表
    #print(cpa)
    print('mpa is : %f' % mpa)
    FWIoU = metric.Frequency_Weighted_Intersection_over_Union()
    print('FWIoU is : %f' % FWIoU)

def fast_hist(a, b, n):  # a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的标签，形状(H×W,)；n是类别数目，实数（在这里为4）
    '''
	核心代码
	'''
    # k是一个一维bool数组，形状(H×W,)；目的是找出标签中需要计算的类别（去掉了背景） k=0或1
    k = (a >= 0) & (a < n)
    # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    # print(a.shape)
    # print(b.shape)
    # print(np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).shape)
    # print(n)
    # print('*'*20)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n,n)


def per_class_iu(hist):  # 分别为每个类别（在这里是4类）计算mIoU，hist的形状(n, n)
    '''
	核心代码
	'''
    # 矩阵的对角线上的值组成的一维数组/矩阵的所有元素之和，返回值形状(n,)
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


# hist.sum(0)=按列相加  hist.sum(1)按行相加

# def label_mapping(input, mapping):#主要是因为CityScapes标签里面原类别太多，这样做把其他类别转换成算法需要的类别（共4类）和背景（标注为255）
#    output = np.copy(input)#先复制一下输入图像
#    for ind in range(len(mapping)):
#        output[input == mapping[ind][0]] = mapping[ind][1]#进行类别映射，最终得到的标签里面之后0-18这19个数加255（背景）
#    return np.array(output, dtype=np.int64)#返回映射的标签


def compute_mIoU(gt_dir, pred_dir, devkit_dir):  # 计算mIoU的函数
    """
    Compute IoU given the predicted colorized images and
    """
    # with open('/home/ubuntu/DeepLab/datasets/VOCdevkit/VOC2012/ImageSets/Segmentation/info.json', 'r') as fp:
    #     # 读取info.json，里面记录了类别数目，类别名称。（我们数据集是VOC2011，相应地改了josn文件）
    #     info = json.load(fp)
    # num_classes = np.int(info['classes'])  # 读取类别数目，这里是20类
    # print('Num classes', num_classes)  # 打印一下类别数目
    # name_classes = np.array(info['label'], dtype=np.str)  # 读取类别名称
    # # mapping = np.array(info['label2train'], dtype=np.int)#读取标签映射方式，详见博客中附加的info.json文件
    # hist = np.zeros((num_classes, num_classes))  # hist初始化为全零，在这里的hist的形状是[20, 20]

    num_classes = 4
    #print('Num classes', num_classes)
    name_classes = ["background","rice","chalky","germ"]
    hist = np.zeros((num_classes, num_classes))

    image_path_list = join(devkit_dir, 'val2.txt')  # 在这里打开记录分割图片名称的txt
    label_path_list = join(devkit_dir, 'val2.txt')  # ground truth和自己的分割结果txt一样
    gt_imgs = open(label_path_list, 'r').read().splitlines()  # 获得验证集标签名称列表
    gt_imgs = [join(gt_dir, x) for x in gt_imgs]  # 获得验证集标签路径列表，方便直接读取
    pred_imgs = open(image_path_list, 'r').read().splitlines()  # 获得验证集图像分割结果名称列表
    pred_imgs = [join(pred_dir, x[:-3]+'txt') for x in pred_imgs]
    # pred_imgs = [join(pred_dir, x) for x in pred_imgs]
    # pred_imgs = [join(pred_dir, x.split('/')[-1]) for x in pred_imgs]#获得验证集图像分割结果路径列表，方便直接读取


    for ind in range(len(gt_imgs)):  # 读取每一个（图片-标签）对
        # pred = np.array(Image.open(pred_imgs[ind]))  # 读取一张图像分割结果，转化成numpy数组
        label = np.array(Image.open(gt_imgs[ind]))  # 读取一张对应的标签，转化成numpy数组
        # print pred.shape
        # print label.shape
        pred = np.loadtxt(pred_imgs[ind]).astype(np.uint8)
        # label = label_mapping(label, mapping)#进行标签映射（因为没有用到全部类别，因此舍弃某些类别），可忽略
        if len(label.flatten()) != 0 and len(pred.flatten()) != 0:
            if len(label.flatten()) != len(pred.flatten()):  # 如果图像分割结果与标签的大小不一样，这张图片就不计算

                print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()),
                                                                                  len(pred.flatten()), gt_imgs[ind],
                                                                                  pred_imgs[ind]))
                continue
            hist += fast_hist(label.flatten(), pred.flatten(), num_classes)  # 对一张图片计算19×19的hist矩阵，并累加
            if ind > 0 and ind % 10 == 0:  # 每计算10张就输出一下目前已计算的图片中所有类别平均的mIoU值
                #print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100 * np.mean(per_class_iu(hist))))
                print(per_class_iu(hist))

    mIoUs = per_class_iu(hist)  # 计算所有验证集图片的逐类别mIoU值
    for ind_class in range(num_classes):  # 逐类别输出一下mIoU值
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))  # 在所有验证集图像上求所有类别平均的mIoU值，计算时忽略NaN值


compute_mIoU('./GT/',
             './output',
             './txt/')  # 执行主函数 三个路径分别为 ‘ground truth’,'自己的实验分割结果'，‘分割图片名称txt文件’
