from torchvision import models

def vgg19_bn(pretrained = True):
    if pretrained == False:
        model = models.vgg19_bn()
    else:
        model = models.vgg19_bn(weights = models.VGG19_BN_Weights)
    return model

def vgg19(pretrained = True):
    if pretrained == False:
        model = models.vgg19()
    else:
        model = models.vgg19(weights = models.VGG19_Weights)
    return model

def vgg16(pretrained = True):
    if pretrained == False:
        model = models.vgg16()
    else:
        model = models.vgg16(weights = models.VGG16_Weights)
    return model

def vgg16_bn(pretrained = True):
    if pretrained == False:
        model = models.vgg16_bn()
    else:
        model = models.vgg16_bn(weights = models.VGG16_BN_Weights)
    return model

