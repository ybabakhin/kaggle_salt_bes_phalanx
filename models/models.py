from models.models_zoo import simple_unet, simple_unet_v2, unet_resnet_50_vgg, unet_inception_resnet_v2, unet_mobilenet, unet_vgg_16, unet_resnet_18, unet_resnet_34, unet_resnet_50, unet_resnet_101, unet_resnet_152, unet_resnet_152_old, unet_resnet_101_do,unet_resnet_101_do_capacity,resnet_34_classification, unet_resnet_50_lovasz, unet_resnet_34_lovasz, unet_resnext_50, unet_resnext_50_lovasz, unet_resnext_101, unet_resnext_101_lovasz
from models.unets import resnet18_fpn, resnet34_fpn, resnet152_fpn, resnet101_fpn, resnet50_fpn, xception_fpn,  densenet_fpn, inception_resnet_v2_fpn, resnet50_fpn_modified

from keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50
from keras.applications.inception_resnet_v2 import preprocess_input as preprocess_input_inception_resnet_v2
from keras.applications.mobilenet import preprocess_input as preprocess_input_mobilenet
from models.classification_models.classification_models.resnet.preprocessing import preprocess_input as preprocess_input_resnet_18_34
from keras.applications.densenet import preprocess_input as preprocess_input_densenet
from keras.applications.xception import preprocess_input as preprocess_input_xception
from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg

def get_model(network, input_shape, freeze_encoder):
    
    if network == 'simple_unet':
        def preprocess(img):
            return img / 255.
        return simple_unet(input_shape), preprocess
    
    elif network == 'simple_unet_v2':
        def preprocess(img):
            return img / 255.
        return simple_unet_v2(input_shape), preprocess
    
    elif network == 'unet_resnet_50_vgg':
        return unet_resnet_50_vgg(input_shape), preprocess_input_resnet50
    
    elif network == 'unet_inception_resnet_v2':
        return unet_inception_resnet_v2(input_shape), preprocess_input_inception_resnet_v2
    
    # 128 minimum
    elif network == 'unet_mobilenet':
        return unet_mobilenet(), preprocess_input_mobilenet
    
    elif network == 'unet_vgg_16':
        return unet_vgg_16(input_shape), preprocess_input_vgg
    
    elif network == 'unet_resnet_18':
        model = unet_resnet_18(input_shape)
        return model, preprocess_input_resnet_18_34
    
    elif network == 'unet_resnet_34':
        model = unet_resnet_34(input_shape,freeze_encoder)
        return model, preprocess_input_resnet_18_34
    
    
    elif network == 'unet_resnet_50':
        model = unet_resnet_50(input_shape,freeze_encoder)
        return model, preprocess_input_resnet_18_34
    
    elif network == 'unet_resnext_50':
        model = unet_resnext_50(input_shape,freeze_encoder)
        return model, preprocess_input_resnet_18_34
    elif network == 'unet_resnext_50_lovasz':
        model = unet_resnext_50_lovasz(input_shape,freeze_encoder)
        return model, preprocess_input_resnet_18_34
    
    
    elif network == 'unet_resnext_101':
        model = unet_resnext_101(input_shape,freeze_encoder)
        return model, preprocess_input_resnet_18_34
    elif network == 'unet_resnext_101_lovasz':
        model = unet_resnext_101_lovasz(input_shape,freeze_encoder)
        return model, preprocess_input_resnet_18_34
    
    
    
    elif network == 'unet_resnet_50_lovasz':
        model = unet_resnet_50_lovasz(input_shape,freeze_encoder)
        return model, preprocess_input_resnet_18_34
    elif network == 'unet_resnet_34_lovasz':
        model = unet_resnet_34_lovasz(input_shape,freeze_encoder)
        return model, preprocess_input_resnet_18_34
    
    elif network == 'unet_resnet_101':
        model = unet_resnet_101(input_shape)
        return model, preprocess_input_resnet_18_34
    elif network == 'unet_resnet_101_do':
        model = unet_resnet_101_do(input_shape)
        return model, preprocess_input_resnet_18_34
    elif network == 'unet_resnet_101_do_capacity':
        model = unet_resnet_101_do_capacity(input_shape)
        return model, preprocess_input_resnet_18_34
    
    
    
    elif network == 'unet_resnet_152':
        model = unet_resnet_152(input_shape,freeze_encoder)
        return model, preprocess_input_resnet_18_34
    elif network == 'unet_resnet_152_old':
        model = unet_resnet_152_old(input_shape,freeze_encoder)
        return model, preprocess_input_resnet_18_34
    

    elif network == 'resnet18_fpn':
        model = resnet18_fpn(input_shape, channels=1, activation="sigmoid", train_base=train_base)
        return model, preprocess_input_resnet_18_34
    
    elif network == 'resnet34_fpn':
        model = resnet34_fpn(input_shape, channels=1, activation="sigmoid", train_base=train_base)
        return model, preprocess_input_resnet_18_34
    
    elif network == 'resnet50_fpn':
        model = resnet50_fpn(input_shape, channels=1, activation="sigmoid", train_base=train_base)
        return model, preprocess_input_resnet50
    
    elif network == 'resnet50_fpn_modified':
        model = resnet50_fpn_modified(input_shape, channels=1, activation="sigmoid", train_base=train_base)
        return model, preprocess_input_resnet50
    
    elif network == 'resnet101_fpn':
        model = resnet101_fpn(input_shape, channels=1, activation="sigmoid")
        return model, preprocess_input_resnet50
    
    elif network == 'resnet152_fpn':
        model = resnet152_fpn(input_shape, channels=1, activation="sigmoid")
        return model, preprocess_input_resnet50
    
    elif network == 'xception_fpn':
        model = xception_fpn(input_shape, channels=1, activation="sigmoid")
        return model, preprocess_input_xception
    
    # 169 minimum
    elif network == 'densenet_fpn':
        model = densenet_fpn(input_shape, channels=1, activation="sigmoid")
        return model, preprocess_input_densenet
    
    elif network == 'inception_resnet_v2_fpn':
        model = inception_resnet_v2_fpn(input_shape, channels=1, activation="sigmoid")
        return model, preprocess_input_inception_resnet_v2
    
    
    elif network == 'classification':
        model = resnet_34_classification(input_shape)
        return model, preprocess_input_resnet_18_34
    
    else:
        raise ValueError('Unknown network ' + network)
        
    return model, preprocess
