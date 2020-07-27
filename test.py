from nets.unet import mobilenet_unet

model = mobilenet_unet(32,input_height=256,input_width=192)
model.summary()

# from nets.uunet import get_unet
# model = get_unet(4,256,256)
# model.summary()
# #
# from nets.unet1 import mobilenet_unet
# model = mobilenet_unet(2,48,48)
# model.summary()
