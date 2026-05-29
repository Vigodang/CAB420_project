from libs import models_UNet
from keras import layers, mixed_precision
mixed_precision.set_global_policy('mixed_float16')

def build_unet(size=128, basef=64, maxf=512, encoder='resnet50', pretrained=True, image_channels=3, use_elevation=True, activation=layers.ReLU):
    return models_UNet.build(
        size=size,
        basef=basef,
        maxf=maxf,
        activation=activation,
        image_channels=image_channels, 
        use_elevation=use_elevation)
