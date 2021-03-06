from model.EfficientNetV2 import EfficientNetV2S
from model.ResNest import resnest
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    UpSampling2D, Activation, BatchNormalization, Conv2D,  Concatenate, LeakyReLU, MaxPooling2D, Input, Flatten, Dense, Dropout, concatenate,
    DepthwiseConv2D,  ZeroPadding2D, Conv2DTranspose, GlobalAveragePooling2D)
from tensorflow.keras.activations import tanh, relu
import tensorflow as tf
from tensorflow.keras.models import Model
MOMENTUM = 0.9
EPSILON = 1e-5
# DECAY = tf.keras.regularizers.L2(l2=0.0001/2)
DECAY = None
# BN = tf.keras.layers.experimental.SyncBatchNormalization
BN = BatchNormalization
CONV_KERNEL_INITIALIZER = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_out", distribution="truncated_normal")
# CONV_KERNEL_INITIALIZER = "he_normal"
activation = 'swish'

def colorization_model(input_shape=(512, 512, 1), classes=3):
    inputs = Input(shape=input_shape, name='image_input')
    bn_momentum = 0.99
    # Encoder branch
    conv1_1 = conv_module(x=inputs, channels=64, kernel_size=3, strides=1, bn_momentum=bn_momentum,
                activation='leakyrelu', dropout=0.0, prefix='conv1_1')
    conv1_2 = conv_module(x=conv1_1, channels=64, kernel_size=3, strides=1, bn_momentum=bn_momentum,
                activation='leakyrelu', dropout=0.0, prefix='conv1_2')
    conv1_pool = MaxPooling2D((2, 2))(conv1_2) # 128x128@64

    conv2_1 = conv_module(x=conv1_pool, channels=128, kernel_size=3, strides=1, bn_momentum=bn_momentum,
                activation='leakyrelu', dropout=0.0, prefix='conv2_1')
    conv2_2 = conv_module(x=conv2_1, channels=128, kernel_size=3, strides=1, bn_momentum=bn_momentum,
                activation='leakyrelu', dropout=0.0, prefix='conv2_2')
    conv2_pool = MaxPooling2D((2, 2))(conv2_2)  # 64x64 @128

    conv3_1 = conv_module(x=conv2_pool, channels=256, kernel_size=3, strides=1, bn_momentum=bn_momentum,
                activation='leakyrelu', dropout=0.0, prefix='conv3_1')
    conv3_2 = conv_module(x=conv3_1, channels=256, kernel_size=3, strides=1, bn_momentum=bn_momentum,
                activation='leakyrelu', dropout=0.0, prefix='conv3_2')
    conv3_pool = MaxPooling2D((2, 2))(conv3_2) # 32x32 @256

    conv4_1 = conv_module(x=conv3_pool, channels=512, kernel_size=3, strides=1, bn_momentum=bn_momentum,
                activation='leakyrelu', dropout=0.0, prefix='conv4_1')
    conv4_2 = conv_module(x=conv4_1, channels=512, kernel_size=3, strides=1, bn_momentum=bn_momentum,
                activation='leakyrelu', dropout=0.0, prefix='conv4_2')
    conv4_pool = MaxPooling2D((2, 2))(conv4_2) # 16x16 @512

    conv5_1 = conv_module(x=conv4_pool, channels=1024, kernel_size=3, strides=1, bn_momentum=bn_momentum,
                activation='leakyrelu', dropout=0.0, prefix='conv5_1')
    conv5_2 = conv_module(x=conv5_1, channels=1024, kernel_size=3, strides=1, bn_momentum=bn_momentum,
                activation='leakyrelu', dropout=0.0, prefix='conv5_2')

    deconv_5 = deconv_module(conv5_2, channels=512, kernel_size=2, strides=2, prefix='deconv_os16')
    decoder = Concatenate()([conv4_2, deconv_5])
    decoder = Dropout(0.4)(decoder)

    decoder = conv_module(x=decoder, channels=512, kernel_size=3, strides=1, bn_momentum=bn_momentum,
                activation='leakyrelu', dropout=0.0, prefix='decoder_4_1')
    decoder = conv_module(x=decoder, channels=512, kernel_size=3, strides=1, bn_momentum=bn_momentum,
                activation='leakyrelu', dropout=0.0, prefix='decoder_4_2')

    decoder = deconv_module(decoder, channels=256, kernel_size=2, strides=2, prefix='deconv_os8')
    decoder = Concatenate()([conv3_2, decoder])
    decoder = Dropout(0.4)(decoder)

    decoder = conv_module(x=decoder, channels=256, kernel_size=3, strides=1, bn_momentum=bn_momentum,
                activation='leakyrelu', dropout=0.0, prefix='decoder_3_1')
    decoder = conv_module(x=decoder, channels=256, kernel_size=3, strides=1, bn_momentum=bn_momentum,
                activation='leakyrelu', dropout=0.0, prefix='decoder_3_2')

    decoder = deconv_module(decoder, channels=128, kernel_size=2, strides=2, prefix='deconv_os4')
    decoder = Concatenate()([conv2_2, decoder])
    decoder = Dropout(0.4)(decoder)

    decoder = conv_module(x=decoder, channels=128, kernel_size=3, strides=1, bn_momentum=bn_momentum,
                          activation='leakyrelu', dropout=0.0, prefix='decoder_2_1')
    decoder = conv_module(x=decoder, channels=128, kernel_size=3, strides=1, bn_momentum=bn_momentum,
                          activation='leakyrelu', dropout=0.0, prefix='decoder_2_2')

    decoder = deconv_module(decoder, channels=64, kernel_size=2, strides=2, prefix='deconv_os2')
    decoder = Concatenate()([conv1_2, decoder])
    decoder = Dropout(0.4)(decoder)

    decoder = conv_module(x=decoder, channels=64, kernel_size=3, strides=1, bn_momentum=bn_momentum,
                          activation='leakyrelu', dropout=0.0, prefix='decoder_1_1')
    decoder = conv_module(x=decoder, channels=64, kernel_size=3, strides=1, bn_momentum=bn_momentum,
                          activation='leakyrelu', dropout=0.0, prefix='decoder_1_2')

    outputs = Conv2D(classes,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    use_bias=True,
                    kernel_initializer='he_normal',
                    padding='same',
                    activation='tanh')(decoder)

    return inputs, outputs


def conv_module(x, channels, kernel_size=3, strides=1,
                bn_momentum=0.8, activation='leakyrelu', dropout=0.4, prefix='name'):
    x = Conv2D(channels,
                     kernel_size=(kernel_size, kernel_size),
                     strides=(strides, strides),
                     use_bias=False,
                     kernel_initializer='he_normal',
                     padding='same',
                     name=prefix+'_conv2d')(x)
    x = BatchNormalization(momentum=bn_momentum, name=prefix+'_bn')(x)

    if dropout != 0.:
        x = Dropout(dropout, name=prefix+'dropout')(x)

    if activation == 'leakyrelu':
        x = Activation(LeakyReLU(0.2), name=prefix+'_activation')(x)
    else:
        x = Activation('relu', name=prefix+'_activation')(x)

    return x

def deconv_module(x, channels, kernel_size=2, strides=2, bn_momentum=0.8, prefix='name'):
    x = Conv2DTranspose(channels,
                      kernel_size=(kernel_size, kernel_size),
                      strides=(strides, strides),
                      use_bias=False,
                      kernel_initializer='he_normal',
                      name=prefix+'_upsampling',
                      padding='same')(x)
    x = BatchNormalization(momentum=bn_momentum, name=prefix+'_bn')(x)
    x = Activation(LeakyReLU(0.2), name=prefix + '_activation')(x)
    # x = UpSampling2D((2, 2), interpolation='bilinear', name=prefix+'_upsampling')(x)
    # x = Conv2D(channels,
    #                  kernel_size=(3, 3),
    #                  strides=(1, 1),
    #                  use_bias=False,
    #                  kernel_initializer='he_normal',
    #                  padding='same',
    #                  name=prefix+'_conv2d')(x)
    # x = BatchNormalization(momentum=bn_momentum, name=prefix+'_bn')(x)
    # x = Activation(LeakyReLU(0.2), name=prefix + '_activation')(x)
    return x

def build_generator(input_shape, output_channels):
    
    

    """U-Net Generator"""

    def conv2d(layer_input, filters, f_size=4, bn=True):
        """Layers used during downsampling"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

    def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        """Layers used during upsampling"""
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = BatchNormalization(momentum=0.8)(u)
        u = Concatenate()([u, skip_input])
        return u

    # Image input
    gf = 64
    d0 = Input(shape=input_shape, name="unet_input")

    # Downsampling
    d1 = conv2d(d0, gf, bn=False)
    d2 = conv2d(d1, gf*2)
    d3 = conv2d(d2, gf*4)
    d4 = conv2d(d3, gf*8)
    d5 = conv2d(d4, gf*8)
    d6 = conv2d(d5, gf*8)
    d7 = conv2d(d6, gf*8)

    # Upsampling
    u1 = deconv2d(d7, d6, gf*8)
    u2 = deconv2d(u1, d5, gf*8)
    u3 = deconv2d(u2, d4, gf*8)
    u4 = deconv2d(u3, d3, gf*4)
    u5 = deconv2d(u4, d2, gf*2)
    u6 = deconv2d(u5, d1, gf)

    u7 = UpSampling2D(size=2)(u6)
    output_img = Conv2D(output_channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)
    
    return d0, output_img

def build_discriminator(image_size=(512, 512, 3), name='discriminator'):
    inputs = Input(shape=image_size, name='image_input')
    init = 'he_normal'
    momentum = 0.8

    conv11 = Conv2D(64,
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    use_bias=False,
                    kernel_initializer=init,
                    padding='same')(inputs)
    # conv11 = BatchNormalization(momentum=momentum)(conv11)
    conv11 = Activation(LeakyReLU(0.2))(conv11)

    conv21 = Conv2D(128,
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    use_bias=False,
                    kernel_initializer=init,
                    padding='same')(conv11)
    conv21 = BatchNormalization(momentum=momentum)(conv21)
    conv21 = Activation(LeakyReLU(0.2))(conv21)

    conv31 = Conv2D(256,
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    use_bias=False,
                    kernel_initializer=init,
                    padding='same')(conv21)
    conv31 = BatchNormalization(momentum=momentum)(conv31)
    conv31 = Activation(LeakyReLU(0.2))(conv31)
    conv31 = Dropout(0.4)(conv31)

    conv41 = Conv2D(512,
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    use_bias=False,
                    kernel_initializer=init,
                    padding='same')(conv31)
    conv41 = BatchNormalization(momentum=momentum)(conv41)
    conv41 = Activation(LeakyReLU(0.2))(conv41)
    conv41 = Dropout(0.4)(conv41)

    conv51 = Conv2D(1024,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    use_bias=False,
                    kernel_initializer=init,
                    padding='same')(conv41)
    conv51 = BatchNormalization(momentum=momentum)(conv51)
    conv51 = Activation(LeakyReLU(0.2))(conv51)
    conv51 = Dropout(0.4)(conv51)

    output = GlobalAveragePooling2D()(conv51) # add gap

    output = Flatten()(output)

    output = Dense(1,
                   kernel_initializer=init,
                   use_bias=True)(output)

    output = Activation('sigmoid')(output)

    return inputs, output


def build_patchDiscriminator(image_size=(512, 512, 3), name='discriminator'):
    def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

    df = 64
    img_A = Input(shape=(512, 512, 1))
    img_B = Input(shape=(512, 512, 2))

    # Concatenate image and conditioning image by channels to produce input
    combined_imgs = Concatenate(axis=-1)([img_A, img_B])

    d1 = d_layer(combined_imgs, df, bn=False)
    d2 = d_layer(d1, df*2)
    d3 = d_layer(d2, df*4)
    d4 = d_layer(d3, df*8)

    validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

    return [img_A, img_B], validity