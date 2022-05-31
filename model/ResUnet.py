from tensorflow.keras.layers import Conv2D, Add, BatchNormalization, Activation, UpSampling2D, Concatenate, LeakyReLU, Input, Conv2DTranspose, Dropout
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model


class ResUNet():
    def __init__(self, image_size):
        self.image_size = image_size
        self.kernel_weights_init = RandomNormal(stddev=0.02)
    
    def bn_act(self, x, act=True):
        x = BatchNormalization()(x)
        if act == True:
            # x = Activation("relu")(x)
            x = LeakyReLU(alpha=0.2)(x)
        return x

    def conv_block(self, x, filters, kernel_size=(4, 4), padding="same", strides=1, dropout=False):
        conv = self.bn_act(x)
        conv = Conv2D(filters, kernel_size, padding=padding, strides=strides, kernel_initializer=self.kernel_weights_init)(conv)
        if dropout:
            conv= Dropout(0.5)(conv)
        return conv

    def stem(self, x, filters, kernel_size=(4, 4), padding="same", strides=1):
        conv = Conv2D(filters, kernel_size, padding=padding, strides=strides, kernel_initializer=self.kernel_weights_init)(x)
        conv = self.conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)
        
        shortcut = Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides, kernel_initializer=self.kernel_weights_init)(x)
        shortcut = self.bn_act(shortcut, act=False)
        
        output = Add()([conv, shortcut])
        return output

    def residual_block(self, x, filters, kernel_size=(4, 4), padding="same", strides=1):
        res = self.conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
        res = self.conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)
        
        shortcut = Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides, kernel_initializer=self.kernel_weights_init)(x)
        shortcut = self.bn_act(shortcut, act=False)
        
        output = Add()([shortcut, res])
        return output

    def upsample_concat_block(self, x, xskip, filters):
        x = Conv2DTranspose(filters, (4,4), strides=(2,2), padding='same', kernel_initializer=self.kernel_weights_init)(x)
        x = BatchNormalization()(x, training=True)
        x = LeakyReLU(alpha=0.2)(x)

        c = Concatenate()([x, xskip])
        return c
    
    
    def res_u_net_generator(self):
        gen_input_shape=(self.image_size[0], self.image_size[1], 1)

        filter_list = [32, 64, 128, 256, 512]
        
        e0 = Input(shape=gen_input_shape)
        
        # Encoder
        e1 = self.stem(x=e0, filters=filter_list[0], strides=1) # 1/1
        e2 = self.residual_block(x=e1, filters=filter_list[1], strides=2) # 1/2
        e3 = self.residual_block(x=e2, filters=filter_list[2], strides=2) # 1/4
        e4 = self.residual_block(x=e3, filters=filter_list[3], strides=2) # 1/8
        e5 = self.residual_block(x=e4, filters=filter_list[4], strides=2) # 1/16
        
        # Bridge
        b0 = self.conv_block(e5, filter_list[4], strides=1, dropout=True) # 1/16
        b1 = self.conv_block(b0, filter_list[4], strides=1, dropout=True) # 1/16
        
        # Decoder
        u1 = self.upsample_concat_block(b1, e4, filters=filter_list[3]) # 1/8
        d1 = self.residual_block(u1, filter_list[3])
        
        u2 = self.upsample_concat_block(d1, e3, filters=filter_list[2]) # 1/4
        d2 = self.residual_block(u2, filter_list[2])
        
        u3 = self.upsample_concat_block(d2, e2, filters=filter_list[1]) # 1/2
        d3 = self.residual_block(u3, filter_list[1])
        
        u4 = self.upsample_concat_block(d3, e1, filters=filter_list[0]) # 1/1
        d4 = self.residual_block(u4, filter_list[0])
        
        outputs = Conv2D(2, (1, 1), padding="same", activation="tanh")(d4)
        model = Model(e0, outputs)
        
        return model
    
    
    def res_discriminator(self):

        input_shape=(self.image_size[0], self.image_size[1], 3)

        
        input = Input(shape=input_shape)
        
        # 256
        d = Conv2D(32, (4,4), strides=(2,2), padding='same', use_bias=True, kernel_initializer=self.kernel_weights_init)(input)
        d = LeakyReLU(alpha=0.2)(d)
        
        # 128
        d = self.residual_block(x=d, filters=64, strides=2)
        
        # 64
        d = self.residual_block(x=d, filters=128, strides=2)
        
        # 32
        d = self.residual_block(x=d, filters=256, strides=2)
        
        # 32
        d = self.residual_block(x=d, filters=512, strides=1)
        
        # for patchGAN
        output = Conv2D(1, (4,4), strides=(1,1), padding='same', use_bias=True, kernel_initializer=self.kernel_weights_init)(d)
    
        # define model
        model = Model(input, output, name='discriminator_model')
        
        return model
        
        
            