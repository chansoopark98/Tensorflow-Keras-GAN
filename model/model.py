from model.EfficientNetV2 import EfficientNetV2S
from tensorflow.keras import layers
from tensorflow.keras.layers import (AveragePooling2D,
    MaxPooling2D, SeparableConv2D, UpSampling2D, Activation, BatchNormalization,
    GlobalAveragePooling2D, Conv2D, Dropout, Concatenate, multiply, Add, concatenate,
    DepthwiseConv2D, Reshape, ZeroPadding2D, Dense, GlobalMaxPooling2D, Permute, Lambda, Subtract)
import tensorflow.keras.backend as K
import tensorflow as tf
import tensorflow_addons as tfa

MOMENTUM = 0.99
EPSILON = 1e-5
# DECAY = tf.keras.regularizers.L2(l2=0.0001/2)
DECAY = None
BN = tf.keras.layers.experimental.SyncBatchNormalization
CONV_KERNEL_INITIALIZER = tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_out", distribution="truncated_normal")
atrous_rates= (6, 12, 18)
activation = 'swish'

def colorization_model(input_shape=(512, 1024, 3), classes=19):
    base = EfficientNetV2S(input_shape=input_shape, pretrained="imagenet")
    base.summary()

    c5 = base.get_layer('add_34').output  # 16x32 256 or get_layer('post_swish') => 확장된 채널 1280
    # c5 = base.get_layer('post_swish').output  # 32x64 256 or get_layer('post_swish') => 확장된 채널 1280
    # c4 = base.get_layer('add_20').output  # 32x64 64
    c3 = base.get_layer('add_7').output  # 64x128 48
    # c2 = base.get_layer('add_6').output  # 128x256 48
    c2 = base.get_layer('add_4').output  # 128x256 48
    """
    for EfficientNetV2S
    32x64 = 'add_34'
    64x128 = 'add_7'
    128x256 = 'add_4'
    """
    features = [c2, c3, c5]

    model_input = base.input
    # model_output, aspp_aux = deepLabV3Plus(features=features, fpn_times=2, activation='swish', mode='deeplabv3+')

    skip1, _, x = features # c1 48 / c2 64

    # Image Feature branch
    shape_before = tf.shape(x)
    b4 = GlobalAveragePooling2D()(x)
    b4_shape = tf.keras.backend.int_shape(b4)
    # from (b_size, channels)->(b_size, 1, 1, channels)
    b4 = Reshape((1, 1, b4_shape[1]))(b4)
    b4 = Conv2D(256, (1, 1), padding='same',
                kernel_regularizer=DECAY,
                use_bias=False, name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = Activation(activation)(b4)
    # upsample. have to use compat because of the option align_corners
    size_before = tf.keras.backend.int_shape(x)
    b4 = tf.keras.layers.experimental.preprocessing.Resizing(
            *size_before[1:3], interpolation="bilinear"
        )(b4)

    # b4 = UpSampling2D(size=(32, 64), interpolation="bilinear")(b4)
    # simple 1x1
    b0 = Conv2D(256, (1, 1), padding='same',
                kernel_regularizer=DECAY,
                use_bias=False, name='aspp0')(x)
    # b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = BN(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = Activation(activation, name='aspp0_activation')(b0)

    b1 = SepConv_BN(x, 256, 'aspp1',
                    rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
    # rate = 12 (24)
    b2 = SepConv_BN(x, 256, 'aspp2',
                    rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
    # rate = 18 (36)
    b3 = SepConv_BN(x, 256, 'aspp3',
                    rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)

    # concatenate ASPP branches & project
    x = Concatenate()([b4, b0, b1, b2, b3])

    x = Conv2D(256, (1, 1), padding='same',
               kernel_regularizer=DECAY,
               use_bias=False, name='concat_projection')(x)
    # x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = BN(name='concat_projection_BN', epsilon=1e-5)(x)
    x = Activation(activation)(x)

    x = Dropout(0.1)(x)

    skip_size = tf.keras.backend.int_shape(skip1)
    x = tf.keras.layers.experimental.preprocessing.Resizing(
        *skip_size[1:3], interpolation="bilinear"
    )(x)

    # x = UpSampling2D((4,4), interpolation='bilinear')(x)

    dec_skip1 = Conv2D(48, (1, 1), padding='same',
                       kernel_regularizer=DECAY,
                       use_bias=False, name='feature_projection0')(skip1)
    # dec_skip1 = BatchNormalization(
    #     name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
    dec_skip1 = BN(
        name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
    dec_skip1 = Activation(activation)(dec_skip1)
    x = Concatenate()([x, dec_skip1])
    x = SepConv_BN(x, 256, 'decoder_conv0',
                   depth_activation=True, epsilon=1e-5)
    x = SepConv_BN(x, 256, 'decoder_conv1',
                   depth_activation=True, epsilon=1e-5)

    x = classifier(x, num_classes=classes, upper=4, name='output')

    return model_input, x


def classifier(x, num_classes=19, upper=4, name=None):
    x = layers.Conv2D(num_classes, 1, strides=1,
                      kernel_regularizer=DECAY,
                      kernel_initializer=CONV_KERNEL_INITIALIZER)(x)
    x = layers.UpSampling2D(size=(upper, upper), interpolation='bilinear', name=name)(x)
    return x


def edge_classifier(x, upper=4, name=None):
    x = layers.Conv2D(1, 1, strides=1,
                      kernel_regularizer=DECAY,
                      kernel_initializer=CONV_KERNEL_INITIALIZER)(x)
    x = Activation('sigmoid')(x)
    x = layers.UpSampling2D(size=(upper, upper), interpolation='bilinear', name=name)(x)
    return x

def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    activation = 'swish'
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
