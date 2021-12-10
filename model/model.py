from model.EfficientNetV2 import EfficientNetV2S
# from model.ResNest import resnest
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    UpSampling2D, Activation, BatchNormalization, Conv2D,  Concatenate, LeakyReLU,
    DepthwiseConv2D,  ZeroPadding2D)
from tensorflow.keras.activations import tanh, relu
import tensorflow as tf

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

    ### Conv high-level feature
    x = Conv3x3(x, 256, rate=1)


    ### Decoder C4 branch ###
    x = Upsampling(x, channel=256)
    x = Concatenate()([x, c4]) #160
    x = Conv1x1(x, channel=256)
    x = SepConv_BN(x, filters=256, prefix="decoder_c4", stride=1, kernel_size=3, rate=1, depth_activation=True)

    ### Decoder C3 branch ###
    x = Upsampling(x, channel=256)
    x = Concatenate()([x, c3]) #64
    x = Conv1x1(x, channel=256)
    x = SepConv_BN(x, filters=256, prefix="decoder_c3", stride=1, kernel_size=3, rate=1, depth_activation=True)

    ### Decoder C2 branch ###
    x = Upsampling(x, channel=256)
    x = Concatenate()([x, c2]) #48
    x = Conv1x1(x, channel=256)
    x = SepConv_BN(x, filters=256, prefix="decoder_c2", stride=1, kernel_size=3, rate=1, depth_activation=True)

    ### Decoder C1 branch ###
    x = Upsampling(x, channel=256)
    x = Concatenate()([x, c1]) # 24
    x = Conv1x1(x, channel=256)
    x = SepConv_BN(x, filters=256, prefix="decoder_c1", stride=1, kernel_size=3, rate=1, depth_activation=True)

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


def Upsampling(x, channel):
    x = UpSampling2D((2, 2), interpolation='nearest')(x)
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
