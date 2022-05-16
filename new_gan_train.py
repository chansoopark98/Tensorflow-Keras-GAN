import tensorflow as tf
import os
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.layers import (
    UpSampling2D, Activation, BatchNormalization, Conv2D,  Concatenate, LeakyReLU, MaxPooling2D, Input, Flatten, Dense, Dropout, concatenate,
    DepthwiseConv2D,  ZeroPadding2D, Conv2DTranspose, GlobalAveragePooling2D)
from tensorflow.keras.activations import tanh, relu
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy, mean_absolute_error, MeanAbsoluteError, MeanSquaredError
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import tensorflow.keras.backend as K
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
        # Input shape
        self.img_rows = 512
        self.img_cols = 512
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------

        # Build the generator
        self.generator = self.build_generator()

        # Input images and their conditioning images
        img_R = Input(shape=(512, 512, 1)) # (512, 512, 2)
        img_GB = Input(shape=(512, 512, 2)) # (512, 512, 1)

        # By conditioning on B generate a fake version of A
        generate_GB = self.generator(img_R) # Input : 512, 512, 1 Output : 512, 512, 2

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([img_R, generate_GB])

        self.combined = tf.keras.Model(inputs=[img_R, img_GB], outputs=[valid, generate_GB])
        self.combined.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)

    def build_generator(self, gen_input_shape=(512, 512, 1)):
        self.initializer = tf.random_normal_initializer(0., 0.02)

        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=(f_size, f_size), strides=2, padding='same',
                        kernel_initializer=self.initializer, use_bias=False)(layer_input)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            d = LeakyReLU(alpha=0.2)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0.5, use_dropout=False):
            """Layers used during upsampling"""
            u = Conv2DTranspose(filters, kernel_size=(f_size, f_size), strides=2, padding='same',
                                kernel_initializer=self.initializer, use_bias=False)(layer_input)
            u = BatchNormalization(momentum=0.8)(u)
            if use_dropout:
                u = Dropout(dropout_rate)(u)
            
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=gen_input_shape) # (256, 256, 1)

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False) # 128
        d2 = conv2d(d1, self.gf*2) # 64
        d3 = conv2d(d2, self.gf*4) # 32
        d4 = conv2d(d3, self.gf*8) # 16
        d5 = conv2d(d4, self.gf*8) # 8
        d6 = conv2d(d5, self.gf*8) # 4
        d7 = conv2d(d6, self.gf*8) # 2
        d8 = conv2d(d7, self.gf*8) # 1

        # Upsampling
        u0 = deconv2d(d8, d7, self.gf*8, use_dropout=True) # 2
        u1 = deconv2d(u0, d6, self.gf*8, use_dropout=True) # 4
        u2 = deconv2d(u1, d5, self.gf*8, use_dropout=True) # 8
        u3 = deconv2d(u2, d4, self.gf*8) # 16
        u4 = deconv2d(u3, d3, self.gf*4) # 32
        u5 = deconv2d(u4, d2, self.gf*2) # 64
        u6 = deconv2d(u5, d1, self.gf) # 128

        output = Conv2DTranspose(2, kernel_size=(4, 4), strides=2, padding='same',
                                kernel_initializer=self.initializer, use_bias=True,
                                activation='tanh')(u6)

        return tf.keras.Model(d0, output)

    def build_discriminator(self):
        self.initializer = tf.random_normal_initializer(0., 0.02)
        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same', kernel_initializer=self.initializer)(layer_input)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            d = LeakyReLU(alpha=0.2)(d)
            
            return d

        img_R = Input(shape=(512, 512, 1)) # (512, 512, 1) R channel
        img_GB = Input(shape=(512, 512, 2)) # (512, 512, 2) GB channel

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_R, img_GB])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same', kernel_initializer=self.initializer)(d4)

        return tf.keras.Model([img_R, img_GB], validity)

    def demo_prepare(self, path):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3)
        return (img)
    
    def train(self):
        EPOCHS = 30
        BATCH_SIZE = 8
        INPUT_SHAPE_GEN = (512, 512, 1)
        

        patch = int(INPUT_SHAPE_GEN[0] / 2**4)
        disc_patch = (patch, patch, 1)

        SCALE_STEP = [512]
        DATASET_DIR ='./datasets'
        CHECKPOINT_DIR = './checkpoints'
        CURRENT_DATE = str(time.strftime('%m%d', time.localtime(time.time())))
        WEIGHTS_GEN = CHECKPOINT_DIR + '/' + CURRENT_DATE + '/GEN'
        WEIGHTS_DIS = CHECKPOINT_DIR + '/' + CURRENT_DATE + '/DIS'
        WEIGHTS_GAN = CHECKPOINT_DIR + '/' + CURRENT_DATE + '/GAN'
        DEMO_OUTPUT = './demo_outputs/' + CURRENT_DATE + '/'
        os.makedirs(DEMO_OUTPUT, exist_ok=True)
        os.makedirs(WEIGHTS_GEN, exist_ok=True)
        os.makedirs(WEIGHTS_DIS, exist_ok=True)
        os.makedirs(WEIGHTS_GAN, exist_ok=True)

        celebA_hq = tfds.load('CustomCelebahq',
                            data_dir=DATASET_DIR, split='train', shuffle_files=True)

        # celebA = tfds.load('CustomCeleba',
        #                        data_dir=DATASET_DIR, split='train', shuffle_files=True)
        #
        # train_data = celebA_hq.concatenate(celebA)
        train_data = celebA_hq

        number_train = train_data.reduce(0, lambda x, _: x + 1).numpy()
        print("학습 데이터 개수", number_train)
        steps_per_epoch = number_train // BATCH_SIZE
        train_data = train_data.shuffle(1024)
        train_data = train_data.padded_batch(BATCH_SIZE)
        train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)

        # prepare validation dataset
        filenames = os.listdir('./demo_images')
        filenames.sort()
        demo_imgs = tf.data.Dataset.list_files('./demo_images/' + '*', shuffle=False)
        demo_test = demo_imgs.map(self.demo_prepare)
        demo_test = demo_test.batch(1)
        demo_steps = len(filenames) // 1



        for steps in range(len(SCALE_STEP)):
            IMAGE_SHAPE = (SCALE_STEP[steps], SCALE_STEP[steps])

            fake_y_dis = tf.zeros((BATCH_SIZE,) + disc_patch)
            real_y_dis = tf.ones((BATCH_SIZE,) + disc_patch)
            # real_y_dis = tf.random.uniform(shape=[(BATCH_SIZE,) + disc_patch], minval=0.9, maxval=1)

            for epoch in range(EPOCHS):
                pbar = tqdm(train_data, total=steps_per_epoch, desc='Batch', leave=True, disable=False)
                batch_counter = 0
                
                dis_res = 0
                index = 0
                for features in pbar:
                    batch_counter += 1
                    # ---------------------
                    #  Train Discriminator
                    # ---------------------
                    # img = tf.cast(features['image'], tf.float32)
                    img = features['image']
                    

                    # data augmentation
                    if tf.random.uniform([], minval=0, maxval=1) > 0.5:
                        img = tf.image.flip_left_right(img)

                    img = tf.image.resize(img, (IMAGE_SHAPE[0], IMAGE_SHAPE[1]), tf.image.ResizeMethod.BILINEAR)

                    img = tf.cast(img, tf.float32) 
                    NORM_RGB = (img / 127.5) - 1
                    
                    R = NORM_RGB[:, :, :, :1]
                    GB = NORM_RGB[:, :, :, 1:]


                    pred_gb = self.generator.predict(R)
        
                    
                    d_real = self.discriminator.train_on_batch([R, GB], real_y_dis)
                    d_fake = self.discriminator.train_on_batch([R, pred_gb], fake_y_dis)
                    dis_res = 0.5 * tf.add(d_fake, d_real)

                    gan_res = self.combined.train_on_batch([R, GB], [real_y_dis, GB])
                    
                    
                    pbar.set_description("Epoch : %d Dis loss: %f Dis ACC: %f Gan loss: %f, MSE loss: %f" % (epoch, dis_res[0],
                                            100 * dis_res[1], gan_res[0], gan_res[2]))
                    
                # if epoch % 5 == 0:
                self.generator.save_weights(WEIGHTS_GEN + str(SCALE_STEP[steps]) + '_'+ str(epoch) + '.h5', overwrite=True)
                self.discriminator.save_weights(WEIGHTS_DIS + str(SCALE_STEP[steps]) + '_'+ str(epoch) + '.h5', overwrite=True)
                self.combined.save_weights(WEIGHTS_GAN + str(SCALE_STEP[steps]) + '_'+ str(epoch) + '.h5', overwrite=True)

                os.makedirs(DEMO_OUTPUT + str(SCALE_STEP[steps]) + '/'+ str(epoch), exist_ok=True)

                # validation
                for img in demo_test:
                    img = tf.image.resize(img, (IMAGE_SHAPE[0], IMAGE_SHAPE[1]), tf.image.ResizeMethod.BILINEAR)

                    img = tf.cast(img, tf.float32) 
                    NORM_RGB = (img / 127.5) - 1
                    
                    R = NORM_RGB[:, :, :, :1]
                    GB = NORM_RGB[:, :, :, 1:]



                    pred_gb = self.generator.predict(R)

                    for i in range(len(pred_gb)):
                        batch_R = R[i]
                        batch_R = (batch_R + 1) * 127.5

                        batch_pred = pred_gb[i]
                        batch_pred = (batch_pred + 1) * 127.5

                        PRED_RGB = tf.concat([batch_R, batch_pred], axis=-1)

                        PRED_RGB = tf.cast(PRED_RGB, tf.uint8)

                        plt.imshow(PRED_RGB)

                        plt.savefig(DEMO_OUTPUT + str(SCALE_STEP[steps]) + '/'+ str(epoch) + '/'+ str(index)+'.png', dpi=200)
                        index +=1

if __name__ == '__main__':
    gan = Pix2Pix()
    gan.train()