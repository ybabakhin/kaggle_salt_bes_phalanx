from models.models_zoo import unet_128, simple_unet, unet_resnet_50
from models.unets import resnet152_fpn, resnet101_fpn, resnet50_fpn, xception_fpn,  densenet_fpn, inception_resnet_v2_fpn

def get_model(network, input_shape):
    if network == 'unet_128':
        return unet_128(input_shape)
    
    elif network == 'simple_unet':
        return simple_unet(input_shape)
    
    elif network == 'unet_resnet_50_224':
        return unet_resnet_50(input_shape)
    
    elif network == 'unet_resnet_50_224_test':
        return unet_resnet_50(input_shape)
    
    elif network == 'resnet152_fpn':
        return resnet152_fpn(input_shape, channels=1, activation="sigmoid")
    elif network == 'resnet101_fpn':
        return resnet101_fpn(input_shape, channels=1, activation="sigmoid")
    elif network == 'resnet50_fpn':
        return resnet50_fpn(input_shape, channels=1, activation="sigmoid")
    elif network == 'resnetv2_fpn':
        return inception_resnet_v2_fpn(input_shape, channels=1, activation="sigmoid")
    elif network == 'densenet169_fpn':
        return densenet_fpn(input_shape, channels=1, activation="sigmoid")
    elif network == 'xception_fpn':
        return xception_fpn(input_shape, channels=1, activation="sigmoid")
    
    else:
        raise ValueError('Unknown network ' + network)