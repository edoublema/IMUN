from tensorflow import keras
from nets.unet import mobilenet_unet
#from nets.uunet import get_unet
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from PIL import Image
from keras import backend as K
import numpy as np
import tensorflow as tf
from visual_callbacks import AccLossPlotter

plotter = AccLossPlotter(graphs=['acc', 'loss'], save_graph=True, save_path='./plot')
NCLASSES = 32
HEIGHT = 256
WIDTH = 192


def generate_arrays_from_file(lines, batch_size):
    # 获取总长度
    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        # 获取一个batch_size大小的数据
        for _ in range(batch_size):
            if i == 0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            # 从文件中读取图像
            img = Image.open("./dataset/jpg" + "/" + name)
            img = img.resize((WIDTH, HEIGHT))
            img = np.array(img)
            img = img / 255
            X_train.append(img)

            name = (lines[i].split(';')[1]).replace("\n", "")
            # 从文件中读取图像 标签
            img = Image.open("./dataset/png" + "/" + name)
            img = img.resize((int(WIDTH), int(HEIGHT)))
            img = np.array(img)
            seg_labels = np.zeros((int(HEIGHT), int(WIDTH), NCLASSES))
            for c in range(NCLASSES):
                seg_labels[:, :, c] = (img[:, :, ] == c).astype(int)
            seg_labels = np.reshape(seg_labels, (-1, NCLASSES))
            Y_train.append(seg_labels)

            # 读完一个周期后重新开始
            i = (i + 1) % n
        yield (np.array(X_train), np.array(Y_train))


"""
简单定义的loss
"""
def loss(y_true, y_pred):

    crossloss = K.categorical_crossentropy(y_true, y_pred)
    loss = 4 * K.sum(crossloss) / HEIGHT / WIDTH
    return loss


def focal_loss(gamma=2., alpha=.5):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return (1/100)*(-K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(K.epsilon() + pt_1)) - K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon())))
        #return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(K.epsilon()+pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
    return focal_loss_fixed


def tversky(y_true, y_pred):
    smooth = 1.
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)


def mixed_loss(y_true,y_pred):
    return loss(y_true,y_pred) + 15*tversky_loss(y_true,y_pred)


if __name__ == "__main__":
    log_dir = "logs/"
    # 获取model
    model = mobilenet_unet(n_classes=NCLASSES, input_height=HEIGHT, input_width=WIDTH)
    #model = get_unet(4,256,256)
    # model.summary()

    # 打开数据集的txt
    with open("./dataset/train.txt", "r") as f:
        lines = f.readlines()

    # 打乱行，这个txt主要用于帮助读取数据来训练
    # 打乱的数据更有利于训练
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    # 90%用于训练，10%用于估计。
    num_val = int(len(lines) * 0.1)
    num_train = len(lines) - num_val

    # 保存的方式，3世代保存一次
    checkpoint = ModelCheckpoint(
        log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss',
        save_weights_only=True,
        save_best_only=False,
        period=1
    )
    # 学习率下降的方式，val_loss3次不下降就下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1
    )
    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=15,
        verbose=1
    )

    # 交叉熵
    model.compile(loss = mixed_loss,
                  optimizer=Adam(lr=1e-4),
                  metrics=['accuracy'])


    # 每次训练四个
    batch_size = 2
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))


    # 开始训练
    model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
                        steps_per_epoch=max(1, num_train // batch_size),
                        validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
                        validation_steps=max(1, num_val // batch_size),
                        epochs=30,
                        initial_epoch=0,
                        #class_weight = {5,20,50,20},
                        callbacks=[plotter,checkpoint,reduce_lr]
                        #callbacks = [checkpoint_period, reduce_lr,early_stopping]
                        )
    #history.loss_plot('epoch')
    model.save_weights(log_dir + 'last.h5')
