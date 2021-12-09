from __future__ import print_function, division
from tensorflow.keras.layers import Input, Dense,  Flatten, Dropout, UpSampling2D, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, LeakyReLU, MaxPooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import tensorflow_io as tfio
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import tensorflow_datasets as tfds

from utils.datasets import Dataset
from model.model_builder import base_model
from model.model import Conv3x3

BATCH_SIZE = 1
EPOCHS = 50
DATASET_DIR = './datasets/'
IMAGE_SIZE = (512, 512)
num_classes = 2

def l1(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))

class GAN():
    def __init__(self):
        self.img_rows = 512
        self.img_cols = 512
        self.channels = 2
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100


        optimizer = Adam(0.0002, 0.5)
        optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')  # tf2.4.1 이전

        # self.options = tf.data.Options()
        # self.options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

        # self.train_dataset_config = Dataset(DATASET_DIR, IMAGE_SIZE, BATCH_SIZE, mode='train',
        #                                dataset='CustomCelebahq')
        # self.train_data = self.train_dataset_config.gan_trainData(self.train_dataset_config.train_data)
        self.train_data = tfds.load('CustomCelebahq',
                               data_dir=DATASET_DIR, split='train[:50%]')
        self.number_train = self.train_data.reduce(0, lambda x, _: x + 1).numpy()
        print("학습 데이터 개수", self.number_train)
        self.train_data = self.train_data.shuffle(1024)
        self.train_data = self.train_data.batch(BATCH_SIZE)
        # self.train_data = self.train_data.prefetch(tf.data.experimental.AUTOTUNE)
        # self.train_data = self.train_data.repeat()

        # self.train_data = self.train_data.with_options(self.options)
        # self.train_data = mirrored_strategy.experimental_distribute_dataset(self.train_data)
        # options = tf.data.Options()
        # options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        # self.train_data = self.train_data.with_options(options)

        self.steps_per_epoch = self.number_train // BATCH_SIZE

        # Build the generator
        self.generator = self.build_generator()
        self.generator.compile(
            optimizer=optimizer,
            loss='mse')
        self.generator.load_weights('test_model.h5', by_name=True)

    def build_generator(self):

        model_input, model_output = base_model(image_size=(512, 512, 1), num_classes=2)
        model = tf.keras.Model(model_input, model_output)
        return model

    def predict(self):
        batch_index=0
        for features in tqdm(self.train_data, total=self.steps_per_epoch):
        # for features in self.train_data:
            # ---------------------
            #  Train Discriminator
            # ---------------------
            img = tf.cast(features['image'], tf.float32)

            img = tf.image.resize(img, (512, 512), tf.image.ResizeMethod.BILINEAR)
            # Generate L,a,b channels image From input RGB data.
            img /= 255.  # input is Float type

            img_lab = tfio.experimental.color.rgb_to_lab(img)
            L_input = img_lab[:, :, :, 0]
            L_input = (L_input / 50.) - 1.
            L_input = tf.expand_dims(L_input, -1)


            # Generate a batch of new images
            gen_imgs = self.generator.predict(L_input)
            L = L_input[0]
            prediction = gen_imgs[0]

            L += 1
            L *= 50.

            a = prediction[:, :, 0]
            a = (a + 1) / 2.
            a *= 255.
            a -= 127.

            b = prediction[:, :, 1]
            b = (b + 1) / 2.
            b *= 255.
            b -= 127.

            L = tf.cast(L, tf.float32)
            a = tf.cast(a, tf.float32)
            b = tf.cast(b, tf.float32)


            a = tf.expand_dims(a, -1)
            b = tf.expand_dims(b, -1)

            output = tf.concat([L, a, b], axis=-1)
            output = tfio.experimental.color.lab_to_rgb(output)
            new_r = output[:, :, 0]
            # new_r *= 255.
            # new_r -= 25.
            # new_r /= 255.

            new_g = output[:, :, 1]
            new_b = output[:, :, 2]
            # new_b *= 255.
            # new_b += 10.
            # new_b /= 255.

            new_r = tf.expand_dims(new_r, -1)
            new_g = tf.expand_dims(new_g, -1)
            new_b = tf.expand_dims(new_b, -1)
            new_output = tf.concat([new_r, new_g, new_b], axis=-1)

            fig = plt.figure()

            ax0 = fig.add_subplot(1, 1, 1)
            ax0.imshow(new_output)
            ax0.set_title('Predict')
            ax0.axis("off")

            plt.savefig('./GAN/images/'  + str(batch_index) + 'output.png', dpi=300)
            # pred = tf.cast(pred, tf.int32)
            plt.show()

            # tf.keras.preprocessing.image.save_img('./GAN/images/' + str(batch_index) + '.png', new_output)
            batch_index += 1


if __name__ == '__main__':
    gan = GAN()
    gan.predict()