import tensorflow as tf
import os
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.layers import (
    UpSampling2D, Activation, BatchNormalization, Conv2D,  Concatenate, LeakyReLU, MaxPooling2D, Input, Flatten, Dense, Dropout, concatenate,
    DepthwiseConv2D,  ZeroPadding2D, Conv2DTranspose, GlobalAveragePooling2D)
from model.ResUnet import ResUNet
from model.Unet import Unet
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import binary_crossentropy, mean_absolute_error, MeanAbsoluteError, MeanSquaredError, BinaryCrossentropy, mean_squared_error
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import tensorflow.keras.backend as K
from tqdm import tqdm
import time
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import tensorflow_io as tfio
# LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal 
# LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4.5.9" python gan_train.py
# LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4.3.0" python multi_gpu_train.py 

class Pix2Pix():
    def __init__(self):
        # Set model prefix name
        self.prefix = 'Final'
        # Input shape
        self.img_rows = 512
        self.img_cols = 512
        self.image_size = (self.img_rows, self.img_cols)
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Set mixed precision
        # self.policy = mixed_precision.Policy('mixed_float16', loss_scale=1024)
        # mixed_precision.set_policy(self.policy)
        
        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64
        self.unet = Unet(self.image_size)
        self.model = self.unet.build_generator()
        self.dis_model = self.build_discriminator()
        self.gen_opt = Adam(lr=0.0002, beta_1=0.5)
        self.disc_opt = Adam(lr=0.0002, beta_1=0.5)
        # opt = mixed_precision.LossScaleOptimizer(opt, loss_scale='dynamic')


    def generator_loss(self, y_true, y_pred):
        """_summary_

        Args:
            y_true (Tensor, float32 (B,H,W,2)): gt
            y_pred (Tensor, float32 ((B,H,W,2)): dl model prediction
        """
        # Calculate mae Loss
        # a_ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true[:, :, :, 1:], y_pred[:, :, :, 1:], max_val=2.0))
        # b_ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true[:, :, :, :1], y_pred[:, :, :, :1], max_val=2.0))
        
        # ssim_loss = 0.5 * (a_ssim_loss + b_ssim_loss)
        
        # alpha = 0.84
        
        # scaled_mae = (1 - alpha) * mae_loss
        # scaled_ssim = alpha * ssim_loss
        
        # total_loss = scaled_mae + scaled_ssim
        loss = mean_absolute_error(y_true, y_pred)
        loss = tf.reduce_mean(loss)

        return loss
    
    def discriminator_loss(self, y_true, y_pred):
        loss = mean_squared_error(y_true, y_pred)
        loss = tf.reduce_mean(loss)
        return loss



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
        model = Model(input_src_image, output, name='discriminator_model')
        model.trainable = True
        return model

    def demo_prepare(self, path):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3)
        return (img)
    
    @tf.function
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
        ab_channel /= 128.       
        
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
        batch_a *= 128.
        batch_b *= 128.
        
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
        
        norm_rgb = (img / 128) - 1
        
        return (l_channel, ab_channel, norm_rgb)
    

    @tf.function
    def predict_data_prepare(self, sample):
        img = sample['image']
        
        
        img = tf.image.resize(img, (self.image_size[0], self.image_size[1]), method=tf.image.ResizeMethod.BILINEAR)
        
        l_channel, ab_channel = self.rgb_to_lab(rgb=img)
        
        norm_rgb = (img / 128) - 1
        
        return (l_channel, ab_channel, norm_rgb)

    
    @tf.function
    def train_step(self, l_channel, ab_channel):
        lab = tf.concat([l_channel, ab_channel], axis=-1)

        fake_y_dis = tf.zeros((self.BATCH_SIZE,) + self.disc_patch)
        real_y_dis = tf.ones((self.BATCH_SIZE,) + self.disc_patch)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

            pred_ab = self.model(l_channel)
            pred_lab = tf.concat([l_channel, pred_ab], axis=-1)

            real_output = self.dis_model(lab)
            fake_output = self.dis_model(pred_lab)

            gen_loss = self.generator_loss(y_true=lab, y_pred=pred_lab)
            disc_real = self.discriminator_loss(y_true=real_y_dis, y_pred=real_output)
            disc_fake = self.discriminator_loss(y_true=fake_y_dis, y_pred=fake_output)
            disc_loss = 0.5 * tf.add(disc_fake, disc_real)

            gradients_of_generator = gen_tape.gradient(gen_loss, self.model.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.dis_model.trainable_variables)

            self.gen_opt.apply_gradients(zip(gradients_of_generator, self.model.trainable_variables))
            self.disc_opt.apply_gradients(zip(gradients_of_discriminator, self.dis_model.trainable_variables))

            return (gen_loss, disc_loss)

    @tf.function
    def distributed_train_step(self, l_channel, ab_channel):
        gen_per_replica_losses = strategy.run(self.train_step, args=(l_channel, ab_channel,))
        gen_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, gen_per_replica_losses,
                         axis=None)
        # dis_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, dis_per_replica_losses,
        #                  axis=None)
        return gen_loss

    def train(self):
        EPOCHS = 100
        self.BATCH_SIZE = 16
        INPUT_SHAPE_GEN = (self.image_size[0], self.image_size[1], 1)
        
        patch = int(INPUT_SHAPE_GEN[0] / 2**4)
        self.disc_patch = (patch, patch, 1)
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
        steps_per_epoch = number_train // self.BATCH_SIZE
        train_data = train_data.shuffle(1024)
        train_data = train_data.map(self.data_augmentation)
        train_data = train_data.padded_batch(self.BATCH_SIZE)
        train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)

        # prepare validation dataset
        filenames = os.listdir('./demo_images')
        filenames.sort()
        demo_imgs = tf.data.Dataset.list_files('./demo_images/' + '*', shuffle=False)
        demo_test = demo_imgs.map(self.demo_prepare)
        demo_test = demo_test.batch(1)
        demo_steps = len(filenames) // 1


        d_real = [0, 0]
        for epoch in range(EPOCHS):
            pbar = tqdm(train_data, total=steps_per_epoch, desc='Batch', leave=True, disable=False)
            
            index = 0

            for l_channel, ab_channel, _ in pbar:
                
                # ---------------------
                #  Train Discriminator
                # ---------------------
                # img = tf.cast(features['image'], tf.float32)

                # gen_loss, disc_loss = self.train_step(l_channel, ab_channel)
                gen_loss = self.distributed_train_step(l_channel, ab_channel)
                pbar.set_description("Epoch : %d Dis loss: %f, Dis ACC: %f" % (epoch, gen_loss[0], gen_loss[1]))
                

                    
if __name__ == '__main__':
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        gan = Pix2Pix()
        gan.train()