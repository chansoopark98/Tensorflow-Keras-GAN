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

        opt = Adam(lr=0.0002, beta_1=0.5)
        # opt = mixed_precision.LossScaleOptimizer(opt, loss_scale='dynamic')

        # Build discriminator
        self.d_model = self.build_discriminator()
        self.d_model.compile(loss=self.discriminator_loss, optimizer=opt, metrics=['accuracy'])

        
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
        # a_ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true[:, :, :, 1:], y_pred[:, :, :, 1:], max_val=2.0))
        # b_ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true[:, :, :, :1], y_pred[:, :, :, :1], max_val=2.0))
        
        # ssim_loss = 0.5 * (a_ssim_loss + b_ssim_loss)
        
        # alpha = 0.84
        
        # scaled_mae = (1 - alpha) * mae_loss
        # scaled_ssim = alpha * ssim_loss
        
        # total_loss = scaled_mae + scaled_ssim
        
        return mae_loss
    
    def discriminator_loss(self, y_true, y_pred):
        # Calculate bce Loss
        dis_loss = tf.reduce_mean(binary_crossentropy(y_true=y_true, y_pred=y_pred, from_logits=True))
        
        # Calculate mse loss
        # dis_loss = mean_squared_error(y_true=y_true, y_pred=y_pred)
        
        return dis_loss
        
    def build_generator(self):
        unet = Unet(image_size=(self.image_size))
        model = unet.build_generator()
        
        # resUnet = ResUNet(image_size=(self.image_size))
        # model = resUnet.res_u_net_generator()
        
        return model


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



        d_real = [0, 0]
        for epoch in range(EPOCHS):
            pbar = tqdm(train_data, total=steps_per_epoch, desc='Batch', leave=True, disable=False)
            
            index = 0

            for l_channel, ab_channel, _ in pbar:
                
                # ---------------------
                #  Train Discriminator
                # ---------------------
                # img = tf.cast(features['image'], tf.float32)
                original_lab = tf.concat([l_channel, ab_channel], axis=-1)
                
                pred_ab = self.gen_model.predict(l_channel)
                
                pred_lab = tf.concat([l_channel, pred_ab], axis=-1)

                # Unfreeze the discriminator
                self.d_model.trainable = True
                
                fake_y_dis = tf.zeros((BATCH_SIZE,) + disc_patch)
                real_y_dis = tf.ones((BATCH_SIZE,) + disc_patch)
        
                if tf.random.uniform([]) < 0.05:
                    real_factor = tf.random.uniform([], minval=0.8, maxval=1.)
                    real_y_dis *= real_factor
                    
                
                d_real = self.d_model.train_on_batch(original_lab, real_y_dis)
                d_fake = self.d_model.train_on_batch(pred_lab, fake_y_dis)
                
                dis_res = 0.5 * tf.add(d_fake, d_real)

                # Freeze the discriminator
                self.d_model.trainable = False
                    
                gan_res = self.gan_model.train_on_batch(l_channel, [real_y_dis, ab_channel])
                
                pbar.set_description("Epoch : %d Dis loss: %f, Dis ACC: %f, Gan loss: %f, Gen loss: %f Gan ACC: %f Gen MAE: %f" % (epoch,
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
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        gan = Pix2Pix()
        gan.train()