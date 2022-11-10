from torchvision import models
"""
torchvision versioning problems
-> python >= 3.7 is required for torchvision >= 0.14 which is the stable one
"""
def vgg19_bn(pretrained = True):
    if pretrained == False:
        model = models.vgg19_bn()
    else:
        try:
            model = models.vgg19_bn(weights = models.VGG19_BN_Weights)
        except:
            model = models.vgg19_bn(pretrained = True)
    return model

def vgg19(pretrained = True):
    if pretrained == False:
        model = models.vgg19()
    else:
        try:
            model = models.vgg19(weights = models.VGG19_Weights)
        except:
            model = models.vgg19(pretrained = True)
    return model

def vgg16(pretrained = True):
    if pretrained == False:
        model = models.vgg16()
    else:
        try:
            model = models.vgg16(weights = models.VGG16_Weights)
        except:
            model = models.vgg16(pretrained = True)
    return model

def vgg16_bn(pretrained = True):
    if pretrained == False:
        model = models.vgg16_bn()
    else:
        try:
            model = models.vgg16_bn(weights = models.VGG16_BN_Weights)
        except:
            model = models.vgg16_bn(pretrained = True)
    return model

