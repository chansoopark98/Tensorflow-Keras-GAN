from re import T
from xmlrpc.client import TRANSPORT_ERROR
import tensorflow as tf
import os
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.layers import (
    UpSampling2D, Activation, BatchNormalization, Conv2D,  Concatenate, LeakyReLU, MaxPooling2D, Input, Flatten, Dense, Dropout, concatenate,
    DepthwiseConv2D,  ZeroPadding2D, Conv2DTranspose, GlobalAveragePooling2D)

from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.activations import tanh, relu
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy, mean_absolute_error, MeanAbsoluteError, MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import tensorflow.keras.backend as K
from torch import batch_norm
from tqdm import tqdm
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import tensorflow_io as tfio
from skimage import color
import time
import numpy as np
# LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal 


class Pix2Pix():
    def __init__(self):
        # Set model prefix name
        self.prefix = 'BCE_RGB_Dis'
        
        # Input shape
        self.img_rows = 512
        self.img_cols = 512
        self.image_size = (self.img_rows, self.img_cols)
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        opt = Adam(lr=0.0002, beta_1=0.5)

        # Build discriminator
        self.d_model = self.build_discriminator()
        self.d_model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])

        
        # Build generator
        self.gen_model = self.build_generator()
    
        self.d_model.trainable = False
        
        input_src_image = Input(shape=(self.image_size[0], self.image_size[1], 1)) # Input = L channel input
        gen_out = self.gen_model(input_src_image) # return ab channel
        
        # (B, H, W, 3)
        generate_lab = tf.concat([input_src_image, gen_out], axis=-1)
        
        # connect the source input and generator output to the discriminator input
        dis_out = self.d_model(generate_lab) #  L, ab channel

        # src image as input, generated image and real/fake classification as output
        self.gan_model = Model(input_src_image, [dis_out, gen_out], name='gan_model')
        
        self.gan_model.compile(loss=[self.discriminator_loss, self.generator_loss], metrics = ['accuracy', 'mae'], optimizer=opt, loss_weights=[1, 100])
        self.gan_model.summary()

    def generator_loss(self, y_true, y_pred):
        """_summary_

        Args:
            y_true (Tensor, float32 (B,H,W,2)): gt
            y_pred (Tensor, float32 ((B,H,W,2)): dl model prediction
        """    
        # Calculate mae Loss
        mae_loss = mean_absolute_error(y_true=y_true, y_pred=y_pred)
        
        return mae_loss        
    
    def discriminator_loss(self, y_true, y_pred):
        # Calculate bce Loss
        bce_loss = tf.reduce_mean(binary_crossentropy(y_true=y_true, y_pred=y_pred, from_logits=True))
        
        return bce_loss
        
    def build_generator(self):
        gen_input_shape=(self.image_size[0], self.image_size[1], 1)

        kernel_weights_init = RandomNormal(stddev=0.02)
        input_src_image = Input(shape=gen_input_shape)

        # encoder model
        e1 = self._encoder_block(input_src_image, 64, batchnorm=False)
        e2 = self._encoder_block(e1, 128)
        e3 = self._encoder_block(e2, 256)
        e4 = self._encoder_block(e3, 512)
        e5 = self._encoder_block(e4, 512)
        e6 = self._encoder_block(e5, 512)
        e7 = self._encoder_block(e6, 512)

        # bottleneck, no batch norm and relu
        b = Conv2D(512, (4,4), strides=(2,2), padding='same', use_bias=True, kernel_initializer=kernel_weights_init)(e7)
        b = Activation('relu')(b)

        # decoder model
        d1 = self._decoder_block(b, e7, 512)
        d2 = self._decoder_block(d1, e6, 512)
        d3 = self._decoder_block(d2, e5, 512)
        d4 = self._decoder_block(d3, e4, 512, dropout=False)
        d5 = self._decoder_block(d4, e3, 256, dropout=False)
        d6 = self._decoder_block(d5, e2, 128, dropout=False)
        d7 = self._decoder_block(d6, e1, 64, dropout=False)

        # output
        g = UpSampling2D()(d7)
        g = Conv2D(filters=2, kernel_size=4, strides=1, padding='same', kernel_initializer=kernel_weights_init)(g)
        
        # g = Conv2DTranspose(2, (4,4), strides=(2,2), padding='same', kernel_initializer=kernel_weights_init)(d7)
    
        out_image = Activation('tanh')(g)
        
        # define model
        model = Model(input_src_image, out_image, name='generator_model')

        return model

    def _decoder_block(self, layer_in, skip_in, n_filters, dropout=True):
        kernel_weights_init = RandomNormal(stddev=0.02)
        # add upsampling layer
        g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', use_bias=False, kernel_initializer=kernel_weights_init)(layer_in)
        # add batch normalization
        g = BatchNormalization(momentum=0.8)(g, training=True)
        if dropout:
            g = Dropout(0.5)(g, training=True)
        # merge with skip connection
        g = Concatenate()([g, skip_in])
        # relu activation
        g = Activation('relu')(g)

        return g

    def _encoder_block(self, layer_in, n_filters, batchnorm=True):
        kernel_weights_init = RandomNormal(stddev=0.02)
        
        if batch_norm == True:
            use_bias = False
        else:
            use_bias = True
        
        # add downsampling layer
        g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', use_bias=use_bias, kernel_initializer=kernel_weights_init)(layer_in)
        if batchnorm:
            g = BatchNormalization(momentum=0.8)(g, training=True)
        # leaky relu activation
        g = LeakyReLU(alpha=0.2)(g)

        return g

    def build_discriminator(self):
        kernel_weights_init = RandomNormal(stddev=0.02)
        src_image_shape = (self.image_size[0], self.image_size[1], 3)
        input_src_image = Input(shape=src_image_shape)
        
        # concatenate images channel-wise
        # merged = Concatenate()([input_src_image, input_target_image])
        merged = input_src_image
        
        # C64
        d = Conv2D(64, (4,4), strides=(2,2), padding='same', use_bias=True, kernel_initializer=kernel_weights_init)(merged)
        d = LeakyReLU(alpha=0.2)(d)
        # C128
        d = Conv2D(128, (4,4), strides=(2,2), padding='same', use_bias=False, kernel_initializer=kernel_weights_init)(d)
        d = BatchNormalization(momentum=0.8)(d)
        d = LeakyReLU(alpha=0.2)(d)
        # C256
        d = Conv2D(256, (4,4), strides=(2,2), padding='same', use_bias=False, kernel_initializer=kernel_weights_init)(d)
        d = BatchNormalization(momentum=0.8)(d)
        d = LeakyReLU(alpha=0.2)(d)
        # C512
        d = Conv2D(512, (4,4), strides=(2,2), padding='same', use_bias=False, kernel_initializer=kernel_weights_init)(d)
        d = BatchNormalization(momentum=0.8)(d)
        d = LeakyReLU(alpha=0.2)(d)
        
        # for patchGAN
        output = Conv2D(1, (4,4), strides=(1,1), padding='same', use_bias=True, kernel_initializer=kernel_weights_init)(d)
    
        # define model
        model = Model(input_src_image, output, name='descriminator_model')
        return model

    def demo_prepare(self, path):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3)
        return (img)
    
    
    def rgb_to_lab(self, rgb):
        """
        Convert to rgb image to lab image

        Args:
            rgb (Tensor): (H, W, 3)

        Returns:
            Normalized lab image
            {
                Value Range
                L : -1 ~ 1
                ab : -1 ~ 1
            }
            L, ab (Tensor): (H, W, 1), (H, W, 2)
        """
        # normalize image 0 ~ 1.
        rgb /= 255. 
        
        # Convert to float32 data type.
        rgb = tf.cast(rgb, tf.float32)
        
        # Convert to rgb to lab
        lab = tfio.experimental.color.rgb_to_lab(rgb)

        l_channel = lab[:, :, 0]
        a_channel = lab[:, :, 1]
        b_channel = lab[:, :, 2]
        
        l_channel = tf.expand_dims(l_channel, axis=-1)
        a_channel = tf.expand_dims(a_channel, axis=-1)
        b_channel = tf.expand_dims(b_channel, axis=-1)

        ab_channel = tf.concat([a_channel, b_channel], axis=-1)
        
        # -1 ~ 1 scaling
        l_channel = (l_channel - 50.) / 50. 
        ab_channel /= 127.       
        
        return l_channel, ab_channel
    
    
    def lab_to_rgb(self, lab, dim=3):
        """
        Convert to lab image to rgb image

        Args:
            lab (Tensor, float32): (H, W, 3)

        Returns:
            {
                Normalized lab image
                Value Range : 0 ~ 1
            }
            RGB (Tensor, float32) :(H, W, 3)
        """      
        if dim == 4:
            batch_l = lab[:, :, :, 0]
            batch_a = lab[:, :, :, 1]
            batch_b = lab[:, :, :, 2]
        else:
            batch_l = lab[:, :, 0]
            batch_a = lab[:, :, 1]
            batch_b = lab[:, :, 2]

        
        batch_l = (batch_l * 50) + 50.
        batch_a *= 127.
        batch_b *= 127.
        
        batch_l = tf.expand_dims(batch_l, axis=-1)
        batch_a = tf.expand_dims(batch_a, axis=-1)
        batch_b = tf.expand_dims(batch_b, axis=-1)
        
        batch_lab = tf.concat([batch_l, batch_a, batch_b], axis=-1)

        rgb = tfio.experimental.color.lab_to_rgb(batch_lab)
        
        return rgb
    
    
    @tf.function
    def data_augmentation(self, sample):
        img = sample['image']
        # data augmentation
        
        if tf.random.uniform([], minval=0, maxval=1) > 0.5:
            img = tf.image.flip_left_right(img)
            
        
        scale = tf.random.uniform([], 0.5, 1.5)
        
        nh = self.image_size[0] * scale
        nw = self.image_size[1] * scale

        img = tf.image.resize(img, (nh, nw), method=tf.image.ResizeMethod.BILINEAR)
        img = tf.image.resize_with_crop_or_pad(img, self.image_size[0], self.image_size[1])
        
        l_channel, ab_channel = self.rgb_to_lab(rgb=img)
        
        norm_rgb = (img / 127.5) - 1
        
        return (l_channel, ab_channel, norm_rgb)

    
    def train(self):
        EPOCHS = 100
        BATCH_SIZE = 8
        INPUT_SHAPE_GEN = (self.image_size[0], self.image_size[1], 1)
        
        patch = int(INPUT_SHAPE_GEN[0] / 2**4)
        disc_patch = (patch, patch, 1)
        DATASET_DIR ='./datasets'
        CHECKPOINT_DIR = './checkpoints'
        CURRENT_DATE = str(time.strftime('%m%d', time.localtime(time.time()))) + '_' + self.prefix
        WEIGHTS_GEN = CHECKPOINT_DIR + '/' + CURRENT_DATE + '/GEN'
        WEIGHTS_DIS = CHECKPOINT_DIR + '/' + CURRENT_DATE + '/DIS'
        WEIGHTS_GAN = CHECKPOINT_DIR + '/' + CURRENT_DATE + '/GAN'
        DEMO_OUTPUT = './demo_outputs/' + CURRENT_DATE + '/'
        os.makedirs(DEMO_OUTPUT, exist_ok=True)
        os.makedirs(WEIGHTS_GEN, exist_ok=True)
        os.makedirs(WEIGHTS_DIS, exist_ok=True)
        os.makedirs(WEIGHTS_GAN, exist_ok=True)

        train_data = tfds.load('CustomCelebahq',
                            data_dir=DATASET_DIR, split='train', shuffle_files=True)

        number_train = train_data.reduce(0, lambda x, _: x + 1).numpy()
        print("학습 데이터 개수", number_train)
        steps_per_epoch = number_train // BATCH_SIZE
        train_data = train_data.shuffle(1024)
        train_data = train_data.map(self.data_augmentation)
        train_data = train_data.padded_batch(BATCH_SIZE)
        train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)

        # prepare validation dataset
        filenames = os.listdir('./demo_images')
        filenames.sort()
        demo_imgs = tf.data.Dataset.list_files('./demo_images/' + '*', shuffle=False)
        demo_test = demo_imgs.map(self.demo_prepare)
        demo_test = demo_test.batch(1)
        demo_steps = len(filenames) // 1

        fake_y_dis = tf.zeros((BATCH_SIZE,) + disc_patch)
        real_y_dis = tf.ones((BATCH_SIZE,) + disc_patch)

        for epoch in range(EPOCHS):
            pbar = tqdm(train_data, total=steps_per_epoch, desc='Batch', leave=True, disable=False)
            batch_counter = 0
            
            dis_res = 0
            index = 0

            for l_channel, ab_channel, norm_rgb in pbar:
                batch_counter += 1
                # ---------------------
                #  Train Discriminator
                # ---------------------
                # img = tf.cast(features['image'], tf.float32)
                
                original_lab = tf.concat([l_channel, ab_channel], axis=-1)
                
                pred_ab = self.gen_model.predict(l_channel)
                
                pred_lab = tf.concat([l_channel, pred_ab], axis=-1)

            
                d_real = self.d_model.train_on_batch(original_lab, real_y_dis)
                d_fake = self.d_model.train_on_batch(pred_lab, fake_y_dis)
                dis_res = 0.5 * tf.add(d_fake, d_real)

                # Freeze the discriminator
                self.d_model.trainable = False
        
                gan_res = self.gan_model.train_on_batch(l_channel, [real_y_dis, ab_channel])
                
                # Unfreeze the discriminator
                self.d_model.trainable = True
                
                pbar.set_description("Epoch : %d Dis loss: %f, Dis ACC: %f Gan loss: %f, Gen loss: %f Gan ACC: %f Gen MAE: %f" % (epoch,
                                            dis_res[0],
                                            dis_res[1],
                                            gan_res[0],
                                            gan_res[1],
                                            gan_res[2],
                                            gan_res[3] * 100))
                
            # if epoch % 5 == 0:
            self.gen_model.save_weights(WEIGHTS_GEN + '_'+ str(epoch) + '.h5', overwrite=True)
            self.d_model.save_weights(WEIGHTS_DIS + '_'+ str(epoch) + '.h5', overwrite=True)
            self.gan_model.save_weights(WEIGHTS_GAN + '_'+ str(epoch) + '.h5', overwrite=True)
            
            save_results_path = DEMO_OUTPUT + '/' + self.prefix + '/'+ str(epoch)
            os.makedirs(save_results_path, exist_ok=True)

            # validation
            for img in demo_test:
                img = tf.image.resize(img, (self.image_size[0], self.image_size[1]), tf.image.ResizeMethod.BILINEAR)
                
                l_channel, _ = self.rgb_to_lab(rgb=img[0])
                
                l_channel = tf.expand_dims(l_channel, axis=0)
                
                pred_ab = self.gen_model.predict(l_channel)
                
                
                pred_lab = tf.concat([l_channel, pred_ab], axis=-1)

                for i in range(len(pred_lab)):
                    
                    rgb = self.lab_to_rgb(lab=pred_lab[i])
                    
                    plt.imshow(rgb)

                    plt.savefig(save_results_path + '/' + str(index)+'.png', dpi=200)
                    index +=1
                    
                    
if __name__ == '__main__':
    gan = Pix2Pix()
    gan.train()