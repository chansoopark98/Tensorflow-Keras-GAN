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
        self.d_model.compile(loss='mse', optimizer=opt, metrics=['accuracy'], loss_weights=[0.5])
        self.d_model.summary()
        
        # Build generator
        self.gen_model = self.build_generator()
    
        self.d_model.trainable = False
        
        input_src_image = Input(shape=(512, 512, 1)) # Input = L channel input
        gen_out = self.gen_model(input_src_image) # return ab channel

        # connect the source input and generator output to the discriminator input
        dis_out = self.d_model([input_src_image, gen_out]) #  L, ab channel

        # src image as input, generated image and real/fake classification as output
        self.gan_model = Model(input_src_image, [dis_out, gen_out], name='gan_model')
        
        self.gan_model.compile(loss=['mse', self.generator_loss], metrics = ['accuracy', 'mae'], optimizer=opt, loss_weights=[1, 100])

    def generator_loss(self, y_true, y_pred):
        """_summary_

        Args:
            y_true (Tensor, float32 (B,H,W,2)): gt
            y_pred (Tensor, float32 ((B,H,W,2)): dl model prediction
        """
        # Set alpha
        alpha = 0.84
        
        # Calculate mae Loss
        mae_loss = mean_absolute_error(y_true=y_true, y_pred=y_pred)
        mae_loss = (1- alpha) * mae_loss
        
        # Calculate multi scale ssim loss
        ms_ssim_loss = tf.image.ssim(img1=y_true, img2=y_pred, max_val=2.0)
        ms_ssim_loss = 1 - tf.reduce_mean(ms_ssim_loss)
        ms_ssim_loss = alpha * ms_ssim_loss
        
        total_loss = ms_ssim_loss + mae_loss
        
        return total_loss        
        
        
    
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
        b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=kernel_weights_init)(e7)
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
        g = Conv2DTranspose(2, (4,4), strides=(2,2), padding='same', kernel_initializer=kernel_weights_init)(d7)
        out_image = Activation('tanh')(g)

        # define model
        model = Model(input_src_image, out_image, name='generator_model')

        return model

    def _decoder_block(self, layer_in, skip_in, n_filters, dropout=True):
        kernel_weights_init = RandomNormal(stddev=0.02)
        # add upsampling layer
        g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=kernel_weights_init)(layer_in)
        # add batch normalization
        g = BatchNormalization()(g, training=True)
        if dropout:
            g = Dropout(0.5)(g, training=True)
        # merge with skip connection
        g = Concatenate()([g, skip_in])
        # relu activation
        g = Activation('relu')(g)

        return g

    def _encoder_block(self, layer_in, n_filters, batchnorm=True):
        kernel_weights_init = RandomNormal(stddev=0.02)
        # add downsampling layer
        g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=kernel_weights_init)(layer_in)
        if batchnorm:
            g = BatchNormalization()(g, training=True)
        # leaky relu activation
        g = LeakyReLU(alpha=0.2)(g)

        return g

    def build_discriminator(self):

        kernel_weights_init = RandomNormal(stddev=0.02)

        src_image_shape = (self.image_size[0], self.image_size[1], 1)
        target_image_shape = (self.image_size[0], self.image_size[1], 2)

        input_src_image = Input(shape=src_image_shape)
        input_target_image = Input(shape=target_image_shape)

        # concatenate images channel-wise
        merged = Concatenate()([input_src_image, input_target_image])
        # C64
        d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=kernel_weights_init)(merged)
        d = LeakyReLU(alpha=0.2)(d)
        # C128
        d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=kernel_weights_init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # C256
        d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=kernel_weights_init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # C512
        d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=kernel_weights_init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        
        # for patchGAN
        output = Conv2D(1, (4,4), strides=(1,1), padding='same', kernel_initializer=kernel_weights_init)(d)
    
        # define model
        model = Model([input_src_image, input_target_image], output, name='descriminator_model')

        return model

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
            # fake_y_dis = tf.zeros((BATCH_SIZE, 1))
            # real_y_dis = tf.ones((BATCH_SIZE, 1))
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
                        
                    
                    scale = tf.random.uniform([], 0.5, 1.5)
                    
                    nh = self.image_size[0] * scale
                    nw = self.image_size[1] * scale

                    img = tf.image.resize(img, (nh, nw), method=tf.image.ResizeMethod.BILINEAR)
                    img = tf.image.resize_with_crop_or_pad(img, self.image_size[0], self.image_size[1])
                    
                    img /= 255. # normalize image 0 ~ 1.
                    img = tf.cast(img, tf.float32)
                    
                    lab = color.rgb2lab(img)

                    l_channel = lab[:, :, :, :1]
                    ab_channel = lab[:, :, :, 1:]

                    # l_channel /= 100. ( 0 ~ 1 scaling )
                    l_channel = (l_channel - 50.) / 50. # ( -1 ~ 1 scaling )
                    
                    ab_channel /= 127.
                    

                    pred_ab = self.gen_model.predict(l_channel)
                    
                    d_real = self.d_model.train_on_batch([l_channel, ab_channel], real_y_dis)
                    d_fake = self.d_model.train_on_batch([l_channel, pred_ab], fake_y_dis)
                    dis_res = 0.5 * tf.add(d_fake, d_real)

                    gan_res = self.gan_model.train_on_batch(l_channel, [real_y_dis, ab_channel])
                    
                    
                    pbar.set_description("Epoch : %d Dis loss: %f, Dis ACC: %f Gan loss: %f, Gen loss: %f Gan ACC: %f Gen MAE: %f" % (epoch,
                                             dis_res[0],
                                             dis_res[1],
                                             gan_res[0],
                                              gan_res[1],
                                              gan_res[2],
                                              gan_res[3] * 100))
                    
                # if epoch % 5 == 0:
                self.gen_model.save_weights(WEIGHTS_GEN + str(SCALE_STEP[steps]) + '_'+ str(epoch) + '.h5', overwrite=True)
                self.d_model.save_weights(WEIGHTS_DIS + str(SCALE_STEP[steps]) + '_'+ str(epoch) + '.h5', overwrite=True)
                self.gan_model.save_weights(WEIGHTS_GAN + str(SCALE_STEP[steps]) + '_'+ str(epoch) + '.h5', overwrite=True)

                os.makedirs(DEMO_OUTPUT + str(SCALE_STEP[steps]) + '/'+ str(epoch), exist_ok=True)

                # validation
                for img in demo_test:
                    img = tf.image.resize(img, (IMAGE_SHAPE[0], IMAGE_SHAPE[1]), tf.image.ResizeMethod.BILINEAR)
                    
                    img /= 255. # normalize image 0 ~ 1.
                    img = tf.cast(img, tf.float32)
                    
                    lab = color.rgb2lab(img)

                    l_channel = lab[:, :, :, :1]
                    ab_channel = lab[:, :, :, 1:]

                    l_channel = (l_channel - 50.) / 50. # ( -1 ~ 1 scaling )
                    ab_channel /= 127.


                    pred_ab = self.gen_model.predict(l_channel)

                    for i in range(len(pred_ab)):

                        # batch_l = l_channel[i] * 100.
                        batch_l = (l_channel[i] * 50) + 50.
                        batch_ab = pred_ab[i]
                        batch_ab *= 127.
                        
                        batch_lab = tf.concat([batch_l, batch_ab], axis=-1)

                        batch_lab = color.lab2rgb(batch_lab)
                        

                        plt.imshow(batch_lab)

                        plt.savefig(DEMO_OUTPUT + str(SCALE_STEP[steps]) + '/'+ str(epoch) + '/'+ str(index)+'.png', dpi=200)
                        index +=1

if __name__ == '__main__':
    gan = Pix2Pix()
    gan.train()