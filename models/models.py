from models.models_zoo import unet_resnext_50, unet_resnext_50_lovasz
from models.segmentation_models.backbones.preprocessing import get_preprocessing


def get_model(network, input_shape, freeze_encoder):
    if network == 'unet_resnext_50':
        model = unet_resnext_50(input_shape, freeze_encoder)
        return model, get_preprocessing('resnext50')
    elif network == 'unet_resnext_50_lovasz':
        model = unet_resnext_50_lovasz(input_shape, freeze_encoder)
        return model, get_preprocessing('resnext50')
    else:
        raise ValueError('Unknown network ' + network)

    return model, preprocess
