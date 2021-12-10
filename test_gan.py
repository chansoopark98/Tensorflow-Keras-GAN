# from __future__ import print_function, division
from tensorflow.keras.layers import Input
from tensorflow.keras.models import  Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np

import tensorflow_datasets as tfds
from model.model_builder import base_model
from model.model import build_discriminator
import tensorflow_io as tfio


def l1(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))

def data_preprocess(sample):
    img = sample['image']
    img = tf.image.resize(img, (512, 512), tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    gray_img = tfio.experimental.color.rgb_to_grayscale(img)
    Gray_3channel = tf.concat([gray_img, gray_img, gray_img], axis=-1)
    gray_ycbcr = tfio.experimental.color.rgb_to_ycbcr(Gray_3channel)
    gray_Y = gray_ycbcr[:, :, 0]
    gray_Y = tf.cast(gray_Y, tf.float32)
    gray_Y = (gray_Y / 127.5) - 1.0
    gray_Y = tf.expand_dims(gray_Y, axis=-1)

    img_YCbCr = tfio.experimental.color.rgb_to_ycbcr(img)
    Cb = img_YCbCr[:, :, 1]
    Cb = tf.cast(Cb, tf.float32)
    Cb = (Cb / 127.5) - 1.0
    Cb = tf.expand_dims(Cb, axis=-1)

    Cr = img_YCbCr[:, :, 2]
    Cr = tf.cast(Cr, tf.float32)
    Cr = (Cr / 127.5) - 1.0
    Cr = tf.expand_dims(Cr, axis=-1)

    CbCr = tf.concat([Cb, Cr], axis=-1)

    return (gray_Y, CbCr)

class GAN():
    def __init__(self):
        """
        Initialize GAN
        """
        self.g_input_shape = (256, 256, 1)
        self.d_input_shape = (256, 256, 2)

        # Build generator
        self.generator = self.build_generator()
        g_optimizer = Adam(learning_rate=0.002, beta_1=0.5, beta_2=0.9)
        self.generator.compile(loss='binary_crossentropy', optimizer=g_optimizer)
        print('Generator Summary...')
        print(self.generator.summary())

        # Build discriminator
        self.discriminator = self.build_discriminator()
        d_optimizer = Adam(learning_rate=0.002, beta_1=0.5, beta_2=0.9)
        self.discriminator.compile(loss='binary_crossentropy', optimizer=d_optimizer, metrics=['accuracy'])
        print('Discriminator Summary...')
        print(self.discriminator.summary())

        gan_input = Input(shape=self.g_input_shape)
        img_color = self.generator(gan_input)
        self.discriminator.trainable = False
        real_or_fake = self.discriminator(img_color)
        self.gan = Model(gan_input, real_or_fake)
        t_optimizer = Adam(learning_rate=0.002, beta_1=0.5, beta_2=0.9)
        self.gan.compile(loss='binary_crossentropy', optimizer=t_optimizer)
        print('\n')
        print('GAN summary...')
        print(self.gan.summary())


    def build_generator(self):
        """
        :return: generator as Keras model
        """
        model_input, model_output = base_model(image_size=self.g_input_shape, num_classes=2)
        generator = tf.keras.Model(model_input, model_output)
        return generator

    def build_discriminator(self):
        """
        :return: discriminator as Keras model.
        """
        discriminator = build_discriminator()
        return discriminator

    def train_discriminator(self, X_train_L, X_train_AB, X_test_L, X_test_AB):
        """
        Function to train the discriminator. Called when discriminator accuracy falls below and a specified threshold.
        """
        generated_images = self.generator.predict(X_train_L)
        X_train = np.concatenate((X_train_AB, generated_images))
        n = len(X_train_L)
        y_train = np.array([[1]] * n + [[0]] * n)
        rand_arr = np.arange(len(X_train))
        np.random.shuffle(rand_arr)
        X_train = X_train[rand_arr]
        y_train = y_train[rand_arr]

        test_generated_images = self.generator.predict(X_test_L)
        X_test = np.concatenate((X_test_AB, test_generated_images))
        n = len(X_test_L)
        y_test = np.array([[1]] * n + [[0]] * n)
        rand_arr = np.arange(len(X_test))
        np.random.shuffle(rand_arr)
        X_test = X_test[rand_arr]
        y_test = y_test[rand_arr]

        self.discriminator.fit(x=X_train, y=y_train, epochs=1)
        metrics = self.discriminator.evaluate(x=X_test, y=y_test)
        print('\n accuracy:',metrics[1])
        if metrics[1] < .90:
            self.train_discriminator(X_train_L, X_train_AB, X_test_L, X_test_AB)

    def train(self, train_data, valid_data, number_trains, number_valids, epochs):
         # self.train_discriminator(X_train_L, X_train_AB, X_test_L, X_test_AB)
        with tf.GradientTape() as tape:
            X_train_L, X_train_AB = train_data
            X_test_L, X_test_AB = valid_data


            g_losses = []
            d_losses = []
            d_acc = []
            X_train = X_train_L
            n = number_trains
            y_train_fake = np.zeros([n,1])
            y_train_real = np.ones([n,1])
            for e in range(epochs):
                #generate images
                np.random.shuffle(X_train)
                generated_images = self.generator.predict(X_train, verbose=1)
                np.random.shuffle(X_train_AB)

                #Train Discriminator
                d_loss  = self.discriminator.fit(train_data,  batch_size=16, epochs=1)
                if e % 3 == 2:
                    noise = np.random.rand(n,256,256,2) * 2 -1
                    d_loss = self.discriminator.fit(x=noise, y=y_train_fake, batch_size=16, epochs=1)
                d_loss = self.discriminator.fit(x=generated_images, y=y_train_fake, batch_size=16, epochs=1)
                d_losses.append(d_loss.history['loss'][-1])
                d_acc.append(d_loss.history['acc'][-1])
                print('d_loss:', d_loss.history['loss'][-1])
                # print("Discriminator Accuracy: ", disc_acc)

                #train GAN on grayscaled images , set output class to colorized
                g_loss = self.gan.fit(x=X_train, y=y_train_real, batch_size=16, epochs=1)

                #Record Losses/Acc
                g_losses.append(g_loss.history['loss'][-1])
                print('Generator Loss: ', g_loss.history['loss'][-1])
                disc_acc = d_loss.history['acc'][-1]

                # Retrain Discriminator if accuracy drops below .8
                if disc_acc < .8 and e < (epochs / 2):
                    self.train_discriminator(X_train_L, X_train_AB, X_test_L, X_test_AB)
                if e % 5 == 4:
                    print(e + 1,"batches done")

if __name__ == '__main__':
    # mirrored_strategy = tf.distribute.MirroredStrategy()
    # with mirrored_strategy.scope():
    train_data = tfds.load('CustomCelebahq',
                                data_dir='./datasets', split='train[5%:]')
    number_train = train_data.reduce(0, lambda x, _: x + 1).numpy()
    print("Train 데이터 개수", number_train)
    train_data = train_data.map(data_preprocess)
    train_data = train_data.shuffle(1024)
    train_data = train_data.batch(16)

    valid_data = tfds.load('CustomCelebahq',
                                data_dir='./datasets', split='train[:5%]')
    number_valid = valid_data.reduce(0, lambda x, _: x + 1).numpy()
    print("Validation 데이터 개수", number_valid)
    valid_data = valid_data.map(data_preprocess)
    valid_data = valid_data.shuffle(1024)
    valid_data = valid_data.batch(16)


    gan = GAN()
    gan.train(train_data=train_data, valid_data=valid_data,
              number_trains=number_train, number_valids=number_valid,
              epochs=10)