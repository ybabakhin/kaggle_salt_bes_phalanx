from keras.applications.vgg16 import VGG16
from keras.engine.training import Model
from models.resnet50_fixed import ResNet50 as ResNet50_fixed
from models.mobile_net_fixed import MobileNet
from models.inception_resnet_v2_fixed import InceptionResNetV2
from keras.applications.vgg16 import VGG16
from keras.layers import *
from models.classification_models.classification_models.resnet.models import ResNet34, ResNet18, ResNet50, ResNet101, ResNet152

def conv_block_simple(prevlayer, filters, prefix, strides=(1, 1)):
    conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides, name=prefix + "_conv")(prevlayer)
    conv = BatchNormalization(name=prefix + "_bn")(conv)
    conv = Activation('relu', name=prefix + "_activation")(conv)
    return conv

def conv_block_simple_no_bn(prevlayer, filters, prefix, strides=(1, 1)):
    conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides, name=prefix + "_conv")(prevlayer)
    conv = Activation('relu', name=prefix + "_activation")(conv)
    return conv

# Probably is used for stacking
def simple_unet(input_shape=(128, 128, 1)):
    img_input = Input(input_shape)
    conv1 = conv_block_simple(img_input, 32, "conv1_1")
    conv1 = conv_block_simple(conv1, 32, "conv1_2")
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool1")(conv1)

    conv2 = conv_block_simple(pool1, 64, "conv2_1")
    conv2 = conv_block_simple(conv2, 64, "conv2_2")
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool2")(conv2)

    conv3 = conv_block_simple(pool2, 128, "conv3_1")
    conv3 = conv_block_simple(conv3, 128, "conv3_2")
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool3")(conv3)

    conv4 = conv_block_simple(pool3, 256, "conv4_1")
    conv4 = conv_block_simple(conv4, 256, "conv4_2")
    conv4 = conv_block_simple(conv4, 256, "conv4_3")

    up5 = concatenate([UpSampling2D()(conv4), conv3], axis=-1)
    conv5 = conv_block_simple(up5, 128, "conv5_1")
    conv5 = conv_block_simple(conv5, 128, "conv5_2")

    up6 = concatenate([UpSampling2D()(conv5), conv2], axis=-1)
    conv6 = conv_block_simple(up6, 64, "conv6_1")
    conv6 = conv_block_simple(conv6, 64, "conv6_2")

    up7 = concatenate([UpSampling2D()(conv6), conv1], axis=-1)
    conv7 = conv_block_simple(up7, 32, "conv7_1")
    conv7 = conv_block_simple(conv7, 32, "conv7_2")

    conv7 = SpatialDropout2D(0.2)(conv7)

    prediction = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv7)
    model = Model(img_input, prediction)
    return model

def simple_unet_v2(input_shape=(128, 128, 1)):
    inputs = Input(shape=input_shape)
    # 128
    
    down1 = conv_block_simple(inputs, 64, "conv1_1")
    down1 = conv_block_simple(down1, 64, "conv1_2")
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool1")(down1)
    # 64

    down2 = conv_block_simple(down1_pool, 128, "conv2_1")
    down2 = conv_block_simple(down2, 128, "conv2_2")
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool2")(down2)
    # 32

    down3 = conv_block_simple(down2_pool, 256, "conv3_1")
    down3 = conv_block_simple(down3, 256, "conv3_2")
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool3")(down3)
    # 16

    down4 = conv_block_simple(down3_pool, 512, "conv4_1")
    down4 = conv_block_simple(down4, 512, "conv4_2")
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool4")(down4)
    # 8
    
    center = conv_block_simple(down4_pool, 1024, "conv5_1")
    center = conv_block_simple(center, 1024, "conv5_2")
    # center

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = conv_block_simple(up4, 512, "conv6_1")
    up4 = conv_block_simple(up4, 512, "conv6_2")
    up4 = conv_block_simple(up4, 512, "conv6_3")
    # 16

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = conv_block_simple(up3, 256, "conv7_1")
    up3 = conv_block_simple(up3, 256, "conv7_2")
    up3 = conv_block_simple(up3, 256, "conv7_3")
    # 32

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = conv_block_simple(up2, 128, "conv8_1")
    up2 = conv_block_simple(up2, 128, "conv8_2")
    up2 = conv_block_simple(up2, 128, "conv8_3")
    # 64

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = conv_block_simple(up1, 64, "conv9_1")
    up1 = conv_block_simple(up1, 64, "conv9_2")
    up1 = conv_block_simple(up1, 64, "conv9_3")
    # 128

    classify = Conv2D(1, (1, 1), activation='sigmoid')(up1)

    model = Model(inputs=inputs, outputs=classify)

    return model

def unet_resnet_50_vgg(input_shape=(128, 128, 1)):
    resnet_base = resnet50_fixed(input_shape=input_shape, include_top=False)
    for l in resnet_base.layers:
        l.trainable = True
    conv1 = resnet_base.get_layer("activation_1").output
    conv2 = resnet_base.get_layer("activation_10").output
    conv3 = resnet_base.get_layer("activation_22").output
    conv4 = resnet_base.get_layer("activation_40").output
    conv5 = resnet_base.get_layer("activation_49").output
    
    up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
    conv6 = conv_block_simple(up6, 256, "conv6_1")
    conv6 = conv_block_simple(conv6, 256, "conv6_2")
    
    up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
    conv7 = conv_block_simple(up7, 192, "conv7_1")
    conv7 = conv_block_simple(conv7, 192, "conv7_2")
    
    up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
    conv8 = conv_block_simple(up8, 128, "conv8_1")
    conv8 = conv_block_simple(conv8, 128, "conv8_2")
    
    up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
    conv9 = conv_block_simple(up9, 64, "conv9_1")
    conv9 = conv_block_simple(conv9, 64, "conv9_2")
    
    vgg = VGG16(input_shape=input_shape, input_tensor=resnet_base.input, include_top=False)
    for l in vgg.layers:
        l.trainable = False
    vgg_first_conv = vgg.get_layer("block1_conv2").output
    up10 = concatenate([UpSampling2D()(conv9), resnet_base.input, vgg_first_conv], axis=-1)
    conv10 = conv_block_simple(up10, 32, "conv10_1")
    conv10 = conv_block_simple(conv10, 32, "conv10_2")
    conv10 = SpatialDropout2D(0.2)(conv10)
    x = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv10)
    model = Model(resnet_base.input, x)
    
    return model

def unet_resnet_18(input_shape):
    resnet_base = ResNet18(input_shape, weights='imagenet', include_top=False)
    
    for l in resnet_base.layers:
        l.trainable = True
    
    conv1 = resnet_base.get_layer("relu0").output
    conv2 = resnet_base.get_layer("stage2_unit1_relu1").output
    conv3 = resnet_base.get_layer("stage3_unit1_relu1").output
    conv4 = resnet_base.get_layer("stage4_unit1_relu1").output
    conv5 = resnet_base.get_layer("relu1").output

    up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
    conv6 = conv_block_simple(up6, 256, "conv6_1")
    conv6 = conv_block_simple(conv6, 256, "conv6_2")

    up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
    conv7 = conv_block_simple(up7, 192, "conv7_1")
    conv7 = conv_block_simple(conv7, 192, "conv7_2")

    up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
    conv8 = conv_block_simple(up8, 128, "conv8_1")
    conv8 = conv_block_simple(conv8, 128, "conv8_2")

    up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
    conv9 = conv_block_simple(up9, 64, "conv9_1")
    conv9 = conv_block_simple(conv9, 64, "conv9_2")

    up10 = concatenate([UpSampling2D()(conv9), resnet_base.input], axis=-1)
    conv10 = conv_block_simple(up10, 32, "conv10_1")
    conv10 = conv_block_simple(conv10, 32, "conv10_2")
    
    conv10 = SpatialDropout2D(0.2)(conv10)
    x = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv10)
    model = Model(resnet_base.input, x)
    return model

def unet_resnet_34(input_shape):
    resnet_base = ResNet34(input_shape, weights='imagenet', include_top=False)
    
    for l in resnet_base.layers:
        l.trainable = True
    
    conv1 = resnet_base.get_layer("relu0").output
    conv2 = resnet_base.get_layer("stage2_unit1_relu1").output
    conv3 = resnet_base.get_layer("stage3_unit1_relu1").output
    conv4 = resnet_base.get_layer("stage4_unit1_relu1").output
    conv5 = resnet_base.get_layer("relu1").output

    up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
    conv6 = conv_block_simple(up6, 256, "conv6_1")
    conv6 = conv_block_simple(conv6, 256, "conv6_2")

    up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
    conv7 = conv_block_simple(up7, 192, "conv7_1")
    conv7 = conv_block_simple(conv7, 192, "conv7_2")

    up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
    conv8 = conv_block_simple(up8, 128, "conv8_1")
    conv8 = conv_block_simple(conv8, 128, "conv8_2")

    up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
    conv9 = conv_block_simple(up9, 64, "conv9_1")
    conv9 = conv_block_simple(conv9, 64, "conv9_2")

    up10 = concatenate([UpSampling2D()(conv9), resnet_base.input], axis=-1)
    conv10 = conv_block_simple(up10, 32, "conv10_1")
    conv10 = conv_block_simple(conv10, 32, "conv10_2")
    
    conv10 = SpatialDropout2D(0.2)(conv10)
    x = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv10)
    model = Model(resnet_base.input, x)
    return model

def unet_resnet_34_do(input_shape):
    resnet_base = ResNet34(input_shape, weights='imagenet', include_top=False)
    
    for l in resnet_base.layers:
        l.trainable = True
    
    conv1 = resnet_base.get_layer("relu0").output
    conv2 = resnet_base.get_layer("stage2_unit1_relu1").output
    conv3 = resnet_base.get_layer("stage3_unit1_relu1").output
    conv4 = resnet_base.get_layer("stage4_unit1_relu1").output
    conv5 = resnet_base.get_layer("relu1").output

#     middle = conv_block_simple(middle, 1024, "conv5_1")
#     middle = conv_block_simple(middle, 1024, "conv5_2")
#     up6_0 = concatenate([UpSampling2D()(middle), conv5], axis=-1)
#     conv6_0 = conv_block_simple(up6_0, 256, "conv6_0_1")
#     conv6_0 = conv_block_simple(conv6_0, 256, "conv6_0_2")
#     up6 = concatenate([UpSampling2D()(conv6_0), conv4], axis=-1)

# capacity to 512 -> 256 -> 128 -> 64 -> 32
    up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
    conv6 = conv_block_simple(up6, 256, "conv6_1")
    conv6 = conv_block_simple(conv6, 256, "conv6_2")

    up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
    conv7 = conv_block_simple(up7, 192, "conv7_1")
    conv7 = conv_block_simple(conv7, 192, "conv7_2")

    up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
    conv8 = conv_block_simple(up8, 128, "conv8_1")
    conv8 = SpatialDropout2D(0.2)(conv8)
    conv8 = conv_block_simple(conv8, 128, "conv8_2")

    up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
    conv9 = conv_block_simple(up9, 64, "conv9_1")
    conv9 = SpatialDropout2D(0.2)(conv9)
    conv9 = conv_block_simple(conv9, 64, "conv9_2")

    up10 = concatenate([UpSampling2D()(conv9), resnet_base.input], axis=-1)
    conv10 = conv_block_simple(up10, 32, "conv10_1")
    conv10 = conv_block_simple(conv10, 32, "conv10_2")
    
    conv10 = SpatialDropout2D(0.4)(conv10)
    x = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv10)
    model = Model(resnet_base.input, x)
    return model

# def unet_resnet_34_do_corrupted(input_shape):
#     resnet_base = ResNet34(input_shape, weights='imagenet', include_top=False)
    
#     for l in resnet_base.layers:
#         l.trainable = True
    
#     conv1 = resnet_base.get_layer("relu0").output
#     conv2 = resnet_base.get_layer("stage2_unit1_relu1").output
#     conv3 = resnet_base.get_layer("stage3_unit1_relu1").output
#     conv4 = resnet_base.get_layer("stage4_unit1_relu1").output
#     conv5 = resnet_base.get_layer("relu1").output

#     #conv5 = conv_block_simple(conv5, 512, "conv5_1")
#     #conv5 = conv_block_simple(conv5, 512, "conv5_2")
    
#     up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
#     #up6 = concatenate([Conv2DTranspose(512, (3, 3), strides=(2, 2), padding="same")(conv5), conv4], axis=-1)
    
#     conv6 = conv_block_simple(up6, 256, "conv6_1")
#     conv6 = conv_block_simple(conv6, 256, "conv6_2")

#     up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
#     #up7 = concatenate([Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same")(conv6), conv3], axis=-1)
#     conv7 = conv_block_simple(up7, 192, "conv7_1")
#     conv7 = conv_block_simple(conv7, 192, "conv7_2")

#     up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
#     #up8 = concatenate([Conv2DTranspose(192, (3, 3), strides=(2, 2), padding="same")(conv7), conv2], axis=-1)
#     conv8 = conv_block_simple(up8, 128, "conv8_1")
#     conv8 = SpatialDropout2D(0.2)(conv8)
#     conv8 = conv_block_simple(conv8, 128, "conv8_2")
    
#     up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
#     #up9 = concatenate([Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(conv8), conv1], axis=-1)
#     conv9 = conv_block_simple(up9, 64, "conv9_1")
#     conv9 = SpatialDropout2D(0.2)(conv9)
#     conv9 = conv_block_simple(conv9, 64, "conv9_2")

#     # Delete first maxpooling in the resnet. And train with 128?
#     # Compare to 256 better
#     # Upconvolutions
    
#     up10 = concatenate([UpSampling2D()(conv9), resnet_base.input], axis=-1)
#     #up10 = concatenate([Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(conv9), resnet_base.input], axis=-1)
#     conv10 = conv_block_simple(up10, 32, "conv10_1")
#     conv10 = conv_block_simple(conv10, 32, "conv10_2")
    
#     # bs == 2: UpSampling2D(size=(32,32))(conv5)
# #     f = concatenate([conv10, UpSampling2D(size=(2,2))(conv9), UpSampling2D(size=(4,4))(conv8),
# #                          UpSampling2D(size=(8,8))(conv7),UpSampling2D(size=(16,16))(conv6)], axis=-1)
    
#     conv10 = SpatialDropout2D(0.4)(conv10)
#     #f = SpatialDropout2D(0.4)(f)
#     x = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv10)
#     model = Model(resnet_base.input, x)
#     return model

def unet_resnet_50(input_shape):
    resnet_base = ResNet50(input_shape, weights='imagenet', include_top=False)
    
    for l in resnet_base.layers:
        l.trainable = True
    
    conv1 = resnet_base.get_layer("relu0").output
    conv2 = resnet_base.get_layer("stage2_unit1_relu2").output
    conv3 = resnet_base.get_layer("stage3_unit1_relu2").output
    conv4 = resnet_base.get_layer("stage4_unit1_relu2").output
    conv5 = resnet_base.get_layer("relu1").output

    up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
    conv6 = conv_block_simple(up6, 256, "conv6_1")
    conv6 = conv_block_simple(conv6, 256, "conv6_2")

    up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
    conv7 = conv_block_simple(up7, 192, "conv7_1")
    conv7 = conv_block_simple(conv7, 192, "conv7_2")

    up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
    conv8 = conv_block_simple(up8, 128, "conv8_1")
    conv8 = conv_block_simple(conv8, 128, "conv8_2")

    up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
    conv9 = conv_block_simple(up9, 64, "conv9_1")
    conv9 = conv_block_simple(conv9, 64, "conv9_2")

    up10 = concatenate([UpSampling2D()(conv9), resnet_base.input], axis=-1)
    conv10 = conv_block_simple(up10, 32, "conv10_1")
    conv10 = conv_block_simple(conv10, 32, "conv10_2")
    
    conv10 = SpatialDropout2D(0.2)(conv10)
    x = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv10)
    model = Model(resnet_base.input, x)
    return model

def unet_resnet_101(input_shape):
    resnet_base = ResNet101(input_shape, weights='imagenet', include_top=False)
    
    for l in resnet_base.layers:
        l.trainable = True
    
    conv1 = resnet_base.get_layer("relu0").output
    conv2 = resnet_base.get_layer("stage2_unit1_relu2").output
    conv3 = resnet_base.get_layer("stage3_unit1_relu2").output
    conv4 = resnet_base.get_layer("stage4_unit1_relu2").output
    conv5 = resnet_base.get_layer("relu1").output

    up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
    conv6 = conv_block_simple(up6, 256, "conv6_1")
    conv6 = conv_block_simple(conv6, 256, "conv6_2")

    up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
    conv7 = conv_block_simple(up7, 192, "conv7_1")
    conv7 = conv_block_simple(conv7, 192, "conv7_2")

    up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
    conv8 = conv_block_simple(up8, 128, "conv8_1")
    conv8 = conv_block_simple(conv8, 128, "conv8_2")

    up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
    conv9 = conv_block_simple(up9, 64, "conv9_1")
    conv9 = conv_block_simple(conv9, 64, "conv9_2")

    up10 = concatenate([UpSampling2D()(conv9), resnet_base.input], axis=-1)
    conv10 = conv_block_simple(up10, 32, "conv10_1")
    conv10 = conv_block_simple(conv10, 32, "conv10_2")
    
    conv10 = SpatialDropout2D(0.2)(conv10)
    x = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv10)
    model = Model(resnet_base.input, x)
    return model

def unet_resnet_101_do(input_shape):
    resnet_base = ResNet101(input_shape, weights='imagenet', include_top=False)
    
    for l in resnet_base.layers:
        l.trainable = True
    
    conv1 = resnet_base.get_layer("relu0").output
    conv2 = resnet_base.get_layer("stage2_unit1_relu2").output
    conv3 = resnet_base.get_layer("stage3_unit1_relu2").output
    conv4 = resnet_base.get_layer("stage4_unit1_relu2").output
    conv5 = resnet_base.get_layer("relu1").output

    conv5 = SpatialDropout2D(0.2)(conv5)
    up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
    conv6 = conv_block_simple(up6, 256, "conv6_1")
    conv6 = conv_block_simple(conv6, 256, "conv6_2")

    conv6 = SpatialDropout2D(0.2)(conv6)
    up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
    conv7 = conv_block_simple(up7, 192, "conv7_1")
    conv7 = conv_block_simple(conv7, 192, "conv7_2")

    conv7 = SpatialDropout2D(0.2)(conv7)
    up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
    conv8 = conv_block_simple(up8, 128, "conv8_1")
    conv8 = conv_block_simple(conv8, 128, "conv8_2")

    conv8 = SpatialDropout2D(0.2)(conv8)
    up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
    conv9 = conv_block_simple(up9, 64, "conv9_1")
    conv9 = conv_block_simple(conv9, 64, "conv9_2")

    conv9 = SpatialDropout2D(0.2)(conv9)
    up10 = concatenate([UpSampling2D()(conv9), resnet_base.input], axis=-1)
    conv10 = conv_block_simple(up10, 32, "conv10_1")
    conv10 = conv_block_simple(conv10, 32, "conv10_2")
    
    conv10 = SpatialDropout2D(0.2)(conv10)
    x = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv10)
    model = Model(resnet_base.input, x)
    return model

# Increase in the center a little bit. Before Upsampling? 3x3 size, m?
def unet_resnet_101_do_capacity(input_shape):
    resnet_base = ResNet101(input_shape, weights='imagenet', include_top=False)
    
    for l in resnet_base.layers:
        l.trainable = True
    
    conv1 = resnet_base.get_layer("relu0").output
    conv2 = resnet_base.get_layer("stage2_unit1_relu2").output
    conv3 = resnet_base.get_layer("stage3_unit1_relu2").output
    conv4 = resnet_base.get_layer("stage4_unit1_relu2").output
    conv5 = resnet_base.get_layer("relu1").output

    conv5 = SpatialDropout2D(0.2)(conv5)
    up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
    conv6 = conv_block_simple(up6, 1024, "conv6_1")
    conv6 = conv_block_simple(conv6, 1024, "conv6_2")

    conv6 = SpatialDropout2D(0.2)(conv6)
    up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
    conv7 = conv_block_simple(up7, 512, "conv7_1")
    conv7 = conv_block_simple(conv7, 512, "conv7_2")

    conv7 = SpatialDropout2D(0.2)(conv7)
    up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
    conv8 = conv_block_simple(up8, 256, "conv8_1")
    conv8 = conv_block_simple(conv8, 256, "conv8_2")

    conv8 = SpatialDropout2D(0.2)(conv8)
    up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
    conv9 = conv_block_simple(up9, 128, "conv9_1")
    conv9 = conv_block_simple(conv9, 128, "conv9_2")

    conv9 = SpatialDropout2D(0.2)(conv9)
    up10 = concatenate([UpSampling2D()(conv9), resnet_base.input], axis=-1)
    conv10 = conv_block_simple(up10, 64, "conv10_1")
    conv10 = conv_block_simple(conv10, 64, "conv10_2")
    
    conv10 = SpatialDropout2D(0.2)(conv10)
    x = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv10)
    model = Model(resnet_base.input, x)
    return model

def unet_resnet_152(input_shape):
    resnet_base = ResNet152(input_shape, weights='imagenet', include_top=False)
    
    for l in resnet_base.layers:
        l.trainable = True
    
    conv1 = resnet_base.get_layer("relu0").output
    conv2 = resnet_base.get_layer("stage2_unit1_relu2").output
    conv3 = resnet_base.get_layer("stage3_unit1_relu2").output
    conv4 = resnet_base.get_layer("stage4_unit1_relu2").output
    conv5 = resnet_base.get_layer("relu1").output

    up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
    conv6 = conv_block_simple(up6, 512, "conv6_1")
    conv6 = conv_block_simple(conv6, 512, "conv6_2")

    up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
    conv7 = conv_block_simple(up7, 256, "conv7_1")
    conv7 = conv_block_simple(conv7, 256, "conv7_2")

    up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
    conv8 = conv_block_simple(up8, 128, "conv8_1")
    conv8 = conv_block_simple(conv8, 128, "conv8_2")

    up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
    conv9 = conv_block_simple(up9, 64, "conv9_1")
    conv9 = conv_block_simple(conv9, 64, "conv9_2")

    up10 = concatenate([UpSampling2D()(conv9), resnet_base.input], axis=-1)
    conv10 = conv_block_simple(up10, 32, "conv10_1")
    conv10 = conv_block_simple(conv10, 32, "conv10_2")
    
    conv10 = SpatialDropout2D(0.2)(conv10)
    x = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv10)
    model = Model(resnet_base.input, x)
    return model

def resnet_34_classification(input_shape):
    resnet_base = ResNet34(input_shape, weights='imagenet', include_top=False)
    
    for l in resnet_base.layers:
        l.trainable = True
    
    base_out = resnet_base.get_layer("relu1").output
    x = GlobalAveragePooling2D()(base_out)
    x = Dropout(0.4)(x)
    x = Dense(1, activation="sigmoid", name="prediction")(x)
    model = Model(resnet_base.input, x)
    
    return model

def unet_mobilenet(input_shape=(128, 128, 1)):
    base_model = MobileNet(include_top=False, input_shape=input_shape)

    conv1 = base_model.get_layer('conv_pw_1_relu').output
    conv2 = base_model.get_layer('conv_pw_3_relu').output
    conv3 = base_model.get_layer('conv_pw_5_relu').output
    conv4 = base_model.get_layer('conv_pw_11_relu').output
    conv5 = base_model.get_layer('conv_pw_13_relu').output
    up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
    conv6 = conv_block_simple(up6, 256, "conv6_1")
    conv6 = conv_block_simple(conv6, 256, "conv6_2")

    up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
    conv7 = conv_block_simple(up7, 256, "conv7_1")
    conv7 = conv_block_simple(conv7, 256, "conv7_2")

    up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
    conv8 = conv_block_simple(up8, 192, "conv8_1")
    conv8 = conv_block_simple(conv8, 128, "conv8_2")

    up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
    conv9 = conv_block_simple(up9, 96, "conv9_1")
    conv9 = conv_block_simple(conv9, 64, "conv9_2")

    up10 = concatenate([UpSampling2D()(conv9), base_model.input], axis=-1)
    conv10 = conv_block_simple(up10, 48, "conv10_1")
    conv10 = conv_block_simple(conv10, 32, "conv10_2")
    conv10 = SpatialDropout2D(0.2)(conv10)
    x = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv10)
    model = Model(base_model.input, x)
    return model

def unet_inception_resnet_v2(input_shape):
    base_model = InceptionResNetV2(include_top=False, input_shape=input_shape)
    conv1 = base_model.get_layer('activation_3').output
    conv2 = base_model.get_layer('activation_5').output
    conv3 = base_model.get_layer('block35_10_ac').output
    conv4 = base_model.get_layer('block17_20_ac').output
    conv5 = base_model.get_layer('conv_7b_ac').output
    up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
    conv6 = conv_block_simple(up6, 256, "conv6_1")
    conv6 = conv_block_simple(conv6, 256, "conv6_2")

    up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
    conv7 = conv_block_simple(up7, 256, "conv7_1")
    conv7 = conv_block_simple(conv7, 256, "conv7_2")

    up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
    conv8 = conv_block_simple(up8, 128, "conv8_1")
    conv8 = conv_block_simple(conv8, 128, "conv8_2")

    up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
    conv9 = conv_block_simple(up9, 64, "conv9_1")
    conv9 = conv_block_simple(conv9, 64, "conv9_2")

    up10 = concatenate([UpSampling2D()(conv9), base_model.input], axis=-1)
    conv10 = conv_block_simple(up10, 48, "conv10_1")
    conv10 = conv_block_simple(conv10, 32, "conv10_2")
    conv10 = SpatialDropout2D(0.4)(conv10)
    x = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv10)
    model = Model(base_model.input, x)
    return model

def unet_vgg_16(input_shape):
    img_input = Input(input_shape)
    vgg16_base = VGG16(input_tensor=img_input, include_top=False)
    for l in vgg16_base.layers:
        l.trainable = True
        
    conv1 = vgg16_base.get_layer("block1_conv2").output
    conv2 = vgg16_base.get_layer("block2_conv2").output
    conv3 = vgg16_base.get_layer("block3_conv3").output
    conv4 = vgg16_base.get_layer("block4_conv3").output
    conv5 = vgg16_base.get_layer("block5_conv3").output
    center = vgg16_base.get_layer("block5_pool").output
   

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([conv5, up4], axis=3)
    up4 = conv_block_simple(up4, 512, "conv6_1")
    up4 = conv_block_simple(up4, 512, "conv6_2")
    up4 = conv_block_simple(up4, 512, "conv6_3")
    # 16

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([conv4, up3], axis=3)
    up3 = conv_block_simple(up3, 256, "conv7_1")
    up3 = conv_block_simple(up3, 256, "conv7_2")
    up3 = conv_block_simple(up3, 256, "conv7_3")
    # 32

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([conv3, up2], axis=3)
    up2 = conv_block_simple(up2, 128, "conv8_1")
    up2 = conv_block_simple(up2, 128, "conv8_2")
    up2 = conv_block_simple(up2, 128, "conv8_3")
    # 64

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([conv2, up1], axis=3)
    up1 = conv_block_simple(up1, 64, "conv9_1")
    up1 = conv_block_simple(up1, 64, "conv9_2")
    up1 = conv_block_simple(up1, 64, "conv9_3")
    # 128
    
    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([conv1, up0], axis=3)
    up0 = conv_block_simple(up0, 64, "conv10_1")
    up0 = conv_block_simple(up0, 64, "conv10_2")
    up0 = conv_block_simple(up0, 64, "conv10_3")
    
    conv10 = SpatialDropout2D(0.2)(up0)
    x = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv10)
    model = Model(img_input, x)
    
    return model

# Unet with dilated convolution in the bottleneck
# https://github.com/lyakaap/Kaggle-Carvana-3rd-Place-Solution
def encoder(x, filters=44, n_block=3, kernel_size=(3, 3), activation='relu'):
    skip = []
    for i in range(n_block):
        x = Conv2D(filters * 2**i, kernel_size, activation=activation, padding='same')(x)
        x = Conv2D(filters * 2**i, kernel_size, activation=activation, padding='same')(x)
        skip.append(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    return x, skip


def bottleneck(x, filters_bottleneck, mode='cascade', depth=6,
               kernel_size=(3, 3), activation='relu'):
    dilated_layers = []
    if mode == 'cascade':  # used in the competition
        for i in range(depth):
            x = Conv2D(filters_bottleneck, kernel_size,
                       activation=activation, padding='same', dilation_rate=2**i)(x)
            dilated_layers.append(x)
        return add(dilated_layers)
    elif mode == 'parallel':  # Like "Atrous Spatial Pyramid Pooling"
        for i in range(depth):
            dilated_layers.append(
                Conv2D(filters_bottleneck, kernel_size,
                       activation=activation, padding='same', dilation_rate=2**i)(x)
            )
        return add(dilated_layers)


def decoder(x, skip, filters, n_block=3, kernel_size=(3, 3), activation='relu'):
    for i in reversed(range(n_block)):
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(filters * 2**i, kernel_size, activation=activation, padding='same')(x)
        x = concatenate([skip[i], x])
        x = Conv2D(filters * 2**i, kernel_size, activation=activation, padding='same')(x)
        x = Conv2D(filters * 2**i, kernel_size, activation=activation, padding='same')(x)
    return x


def get_dilated_unet(
        input_shape=(128, 128, 3),
        mode='cascade',
        filters=44,
        n_block=3
):
    inputs = Input(input_shape)
    
    enc, skip = encoder(inputs, filters, n_block)
    bottle = bottleneck(enc, filters_bottleneck=filters * 2**n_block, mode=mode)
    dec = decoder(bottle, skip, filters, n_block)
    classify = Conv2D(1, (1, 1), activation='sigmoid')(dec)

    model = Model(inputs=inputs, outputs=classify)

    return model
