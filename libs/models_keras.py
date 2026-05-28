from libs import models_UNet

def build_unet(size=300, basef=64, maxf=512, encoder='resnet50', pretrained=True, image_channels=3, use_elevation=True):
    return models_UNet.build(
        size=size,
        basef=basef,
        maxf=maxf,
        encoder=encoder,
        pretrained=pretrained, 
        image_channels=image_channels, 
        use_elevation=use_elevation)