from model.model_builder import build_dis, build_gen
import tensorflow as tf
import os
from tensorflow.keras.layers import Input, concatenate
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
l_cent = 50.
l_norm = 100.
ab_norm = 110.

def eacc(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

def ssim_loss(y_true, y_pred):
    mse_loss = MeanSquaredError()(y_true, y_pred)  

    y_true += 1.
    y_pred += 1.
    
    ssim_loss = (1 - tf.image.ssim_multiscale(y_pred, y_true, 2.0))
    ssim_loss = tf.reduce_mean(ssim_loss) * 0.84
    
    
    return mse_loss + ssim_loss

def mae_loss(y_true, y_pred):
    return MeanAbsoluteError()(y_true, y_pred)
    
def l1(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))

def create_model_gen(input_shape, output_channels):
    model_input, model_output = build_gen(image_size=input_shape, output_channels=output_channels)

    model = tf.keras.Model(model_input, model_output)
    return model

def create_model_dis(input_shape):
    model_input, model_output = build_dis(image_size=input_shape)

    model = tf.keras.Model(model_input, model_output)
    return model

def create_model_gan(input_shape, generator, discriminator):
    input = Input(input_shape)

    gen_out = generator(input)
    gen_out = concatenate([input, gen_out], axis=3)
    dis_out = discriminator(gen_out)
    # dis_out = discriminator(gen_out)

    model = tf.keras.Model(inputs=[input], outputs=[dis_out, gen_out], name='dcgan')
    return model

def create_models(input_shape_gen, input_shape_dis, output_channels, lr, momentum, loss_weights):

    optimizer = Adam(learning_rate=lr, beta_1=momentum)
    optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')  # tf2.4.1 이전

    model_dis = create_model_dis(input_shape=input_shape_dis)
    model_dis.compile(
        loss=binary_crossentropy,
        optimizer=optimizer,
        metrics=['accuracy']
    )

    model_dis.trainable = False

    model_gen = create_model_gen(input_shape=input_shape_gen, output_channels=output_channels)
    model_gen.compile(loss=ssim_loss, optimizer=optimizer)

    model_gan = create_model_gan(input_shape=input_shape_gen, generator=model_gen, discriminator=model_dis)
    model_gan.compile(
        loss=[binary_crossentropy, 'mae'],
        metrics=['accuracy', 'mse'],
        loss_weights=loss_weights,
        optimizer=optimizer
    )

    model_gan.summary()

    return model_gen, model_dis, model_gan

def demo_prepare(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    return (img)

if __name__ == '__main__':
    EPOCHS = 30
    BATCH_SIZE = 8
    LEARNING_RATE = 0.0002
    MOMENTUM = 0.5
    LAMBDA1 = 1
    LAMBDA2 = 100
    INPUT_SHAPE_GEN = (512, 512, 1)
    INPUT_SHAPE_DIS = (512, 512, 3)

    patch = int(INPUT_SHAPE_GEN[0] / 2**4)
    disc_patch = (patch, patch, 1)

    SCALE_STEP = [512]
    GEN_OUTPUT_CHANNEL = 2
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

    model_gen, model_dis, model_gan = create_models(
        input_shape_gen=INPUT_SHAPE_GEN,
        input_shape_dis=INPUT_SHAPE_DIS,
        output_channels=GEN_OUTPUT_CHANNEL,
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        loss_weights=[LAMBDA1, LAMBDA2])

    # model_gen.load_weights(WEIGHTS_GEN + '.h5')
    # model_dis.load_weights(WEIGHTS_DIS + '.h5')
    # model_gan.load_weights(WEIGHTS_GAN + '.h5')

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
    demo_test = demo_imgs.map(demo_prepare)
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
            toggle = True
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


                pred_gb = model_gen.predict(R)
                
                fake_x_dis = tf.concat([R, pred_gb], axis=-1)
                real_x_dis = NORM_RGB
                
                d_real = model_dis.train_on_batch(real_x_dis, real_y_dis)
                d_fake = model_dis.train_on_batch(fake_x_dis, fake_y_dis)

                dis_res = 0.5 * tf.add(d_fake, d_real)

                x_output = NORM_RGB

                gan_res = model_gan.train_on_batch(R, [real_y_dis, x_output])
                model_dis.trainable = True
                
                pbar.set_description("Epoch : %d Dis loss: %f Gan total: %f Gan loss: %f Gan L1: %f ACC: %f MSE: %f" % (epoch, dis_res[0],
                                        gan_res[0], gan_res[1], gan_res[2], gan_res[5], gan_res[6]))


            # if epoch % 5 == 0:
            model_gen.save_weights(WEIGHTS_GEN + str(SCALE_STEP[steps]) + '_'+ str(epoch) + '.h5', overwrite=True)
            model_dis.save_weights(WEIGHTS_DIS + str(SCALE_STEP[steps]) + '_'+ str(epoch) + '.h5', overwrite=True)
            model_gan.save_weights(WEIGHTS_GAN + str(SCALE_STEP[steps]) + '_'+ str(epoch) + '.h5', overwrite=True)

            os.makedirs(DEMO_OUTPUT + str(SCALE_STEP[steps]) + '/'+ str(epoch), exist_ok=True)

            # validation
            for img in demo_test:
                img = tf.image.resize(img, (IMAGE_SHAPE[0], IMAGE_SHAPE[1]), tf.image.ResizeMethod.BILINEAR)

                img = tf.cast(img, tf.float32) 
                NORM_RGB = (img / 127.5) - 1
                
                R = NORM_RGB[:, :, :, :1]
                GB = NORM_RGB[:, :, :, 1:]



                pred_gb = model_gen.predict(R)

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