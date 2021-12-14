from model.EfficientNetV2 import EfficientNetV2S
# from model.ResNest import resnest
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    UpSampling2D, Activation, BatchNormalization, Conv2D,  Concatenate, LeakyReLU, MaxPooling2D, Input, Flatten, Dense, Dropout,
    DepthwiseConv2D,  ZeroPadding2D)
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

def colorization_model(input_shape=(512, 512, 1), classes=2):

    # base.summary()

    """ EfficientNetV2S """
    base = EfficientNetV2S(input_shape=input_shape, pretrained=None)
    c1 = base.get_layer('add_1').output  # 1/2 @24
    c2 = base.get_layer('add_4').output  # 1/4 @48
    c3 = base.get_layer('add_7').output  # 1/8 @64
    c4 = base.get_layer('add_20').output  # 1/16 @160
    x = base.get_layer('add_34').output  # 1/32 @256


    """ ResNest-101"""
    # base = resnest.resnest101(input_shape=input_shape, include_top=False, weights="imagenet", input_tensor=None,
    #                            classes=1000)
    # c1 = base.get_layer('stem_act3').output  # 1/2 @ 128
    # c2 = base.get_layer('stage1_block3_shorcut_act').output  # 1/4 @ 256
    # c3 = base.get_layer('stage2_block4_shorcut_act').output  # 1/8 @ 512
    # c4 = base.get_layer('stage3_block23_shorcut_act').output  # 1/16 @ 1024
    # x = base.get_layer('stage4_block3_shorcut_act').output  # 1/32 @2048

    model_input = base.input

    ### Decoder C4 branch ###
    x = Upsampling(x, channel=256)
    x = Concatenate()([x, c4]) #160
    x = SepConv_BN(x, filters=256, prefix="decoder_c4_1", stride=1, kernel_size=3, rate=1, depth_activation=True)
    x = SepConv_BN(x, filters=256, prefix="decoder_c4_2", stride=1, kernel_size=3, rate=1, depth_activation=True)

    ### Decoder C3 branch ###
    x = Upsampling(x, channel=256)
    x = Concatenate()([x, c3]) #64
    x = SepConv_BN(x, filters=224, prefix="decoder_c3_1", stride=1, kernel_size=3, rate=1, depth_activation=True)
    x = SepConv_BN(x, filters=224, prefix="decoder_c3_2", stride=1, kernel_size=3, rate=1, depth_activation=True)

    ### Decoder C2 branch ###
    x = Upsampling(x, channel=256)
    x = Concatenate()([x, c2]) #48
    x = SepConv_BN(x, filters=128, prefix="decoder_c2_1", stride=1, kernel_size=3, rate=1, depth_activation=True)
    x = SepConv_BN(x, filters=128, prefix="decoder_c2_2", stride=1, kernel_size=3, rate=1, depth_activation=True)

    ### Decoder C1 branch ###
    x = Upsampling(x, channel=256)
    x = Concatenate()([x, c1]) # 24
    x = SepConv_BN(x, filters=64, prefix="decoder_c1_1", stride=1, kernel_size=3, rate=1, depth_activation=True)
    x = SepConv_BN(x, filters=64, prefix="decoder_c1_2", stride=1, kernel_size=3, rate=1, depth_activation=True)

    ### Classifier ###
    x = classifier(x, 2)

    ### Decoder output branch ###
    model_output = Upsampling(x, channel=2)

    # model_output = green
    return model_input, model_output


def classifier(x, num_classes=2, upper=2, name=None):
    x = layers.Conv2D(num_classes, 3, strides=1, padding='same',
                      kernel_initializer=CONV_KERNEL_INITIALIZER, name="classifier")(x)
    x = tanh(x)
    return x



def Conv1x1(x, channel, epsilon=1e-3):
    x = Conv2D(channel, (1, 1), padding='same',
                       kernel_regularizer=DECAY,
                        kernel_initializer=CONV_KERNEL_INITIALIZER,
                       use_bias=False)(x)
    x = BN(axis=-1, epsilon=epsilon)(x)
    x = Activation(activation)(x)
    return x

def Conv3x3(x, channel, rate, activation='swish'):
    x = Conv2D(channel, (3, 3), padding='same', dilation_rate=(rate, rate),
                       kernel_initializer=CONV_KERNEL_INITIALIZER,
                        kernel_regularizer=DECAY,
                       use_bias=False)(x)
    x = BN(axis=-1, epsilon=1e-5)(x)
    x = Activation(activation)(x)

    return x

def create_conv(filters, kernel_size, inputs, name=None, bn=True, dropout=0., padding='same', activation='relu'):
    conv = Conv2D(filters, kernel_size, padding=padding,
                  kernel_initializer='he_normal', name=name)(inputs)

    if bn:
        conv = BatchNormalization()(conv)

    if activation == 'relu':
        conv = Activation(activation)(conv)
    elif activation == 'leakyrelu':
        conv = LeakyReLU()(conv)

    if dropout != 0:
        conv = Dropout(dropout)(conv)

    return conv

def Upsampling(x, channel):
    x = UpSampling2D((2, 2), interpolation='bilinear')(x)
    return x



def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = Activation(activation)(x)
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        kernel_regularizer=DECAY,
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    # x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    x = BN(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation(activation)(x)
    x = Conv2D(filters, (1, 1), padding='same',
               kernel_regularizer=DECAY,
               use_bias=False, name=prefix + '_pointwise')(x)
    # x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    x = BN(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation(activation)(x)

    return x

def build_discriminator(image_size=(512, 512, 2), name='discriminator'):

    inputs = Input(shape=image_size)
    conv1 = create_conv(64, (3, 3), inputs, 'conv1', activation='leakyrelu', dropout=.8)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = create_conv(128, (3, 3), pool1, 'conv2', activation='leakyrelu', dropout=.8)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = create_conv(256, (3, 3), pool2, 'conv3', activation='leakyrelu', dropout=.8)
    pool3 = MaxPooling2D((2, 2))(conv3)

    conv4 = create_conv(512, (3, 3), pool3, 'conv4', activation='leakyrelu', dropout=.8)
    pool4 = MaxPooling2D((2, 2))(conv4)

    conv5 = create_conv(512, (3, 3), pool4, 'conv5', activation='leakyrelu', dropout=.8)

    flat = Flatten()(conv5)
    dense6 = Dense(1, activation='sigmoid')(flat)

    return inputs, dense6