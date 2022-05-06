from model.model_builder import build_dis, build_gen
import tensorflow as tf
import os
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy, mean_absolute_error
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import tensorflow.keras.backend as K
from tqdm import tqdm
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import tensorflow_io as tfio
from skimage import color
import time
# LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4.3.0"


def eacc(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

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
    # dis_out = discriminator(concatenate([gen_out, input], axis=3))
    dis_out = discriminator(gen_out)

    model = tf.keras.Model(inputs=[input], outputs=[dis_out, gen_out], name='dcgan')
    return model

def create_models(input_shape_gen, input_shape_dis, output_channels, lr, momentum, loss_weights):

    optimizer = Adam(learning_rate=lr, beta_1=momentum)
    optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')  # tf2.4.1 이전

    model_gen = create_model_gen(input_shape=input_shape_gen, output_channels=output_channels)
    model_gen.compile(loss=mean_absolute_error, optimizer=optimizer)

    model_dis = create_model_dis(input_shape=input_shape_dis)
    model_dis.trainable = False

    model_gan = create_model_gan(input_shape=input_shape_gen, generator=model_gen, discriminator=model_dis)
    model_gan.compile(
        loss=[binary_crossentropy, l1],
        metrics=[eacc, 'accuracy'],
        loss_weights=loss_weights,
        optimizer=optimizer
    )

    model_gan.summary()

    model_dis.trainable = True
    model_dis.compile(loss=binary_crossentropy, optimizer=optimizer)

    return model_gen, model_dis, model_gan

def rgb2gray(img):
    gray = color.rgb2gray(img)
    gray = tf.cast(gray, tf.float32)
    gray /= 127.5
    gray -= 1.
    return gray

def rgb2lab(img):
    img /= 255.
    lab = tfio.experimental.color.rgb_to_lab(img)

    l = lab[:, :, :, 0]
    l = (l - 50.) / 100.

    a = lab[:, :, :, 1]
    a = a / 110.

    b = lab[:, :, :, 2]
    b = b / 110.

    l = tf.expand_dims(l, axis=-1)
    a = tf.expand_dims(a, axis=-1)
    b = tf.expand_dims(b, axis=-1)

    lab = tf.concat([l, a, b], axis=-1)

    return lab

def demo_prepare(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    return (img)

if __name__ == '__main__':
    EPOCHS = 30
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0002
    MOMENTUM = 0.5
    LAMBDA1 = 1
    LAMBDA2 = 100
    INPUT_SHAPE_GEN = (512, 512, 1)
    INPUT_SHAPE_DIS = (512, 512, 3)
    SCALE_STEP = [256]
    GEN_OUTPUT_CHANNEL = 3
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

        fake_y_dis = tf.zeros((shape[0], 1))
        real_y_dis = tf.random.uniform(shape=[shape[0]], minval=0.9, maxval=1)

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
                shape = img.shape

                # data augmentation
                if tf.random.uniform([], minval=0, maxval=1) > 0.5:
                    img = tf.image.flip_left_right(img)

                img = tf.image.resize(img, (IMAGE_SHAPE[0], IMAGE_SHAPE[1]), tf.image.ResizeMethod.BILINEAR)

                img /= 255.

                r_channel = img[:, :, :, 0]

                fake_x_dis = model_gen.predict(r_channel)

                real_x_dis = img
                d_real = model_dis.train_on_batch(real_x_dis, real_y_dis)
                d_fake = model_dis.train_on_batch(fake_x_dis, fake_y_dis)

                dis_res = 0.5 * tf.add(d_fake, d_real)

                model_dis.trainable = False
                x_gen = r_channel

                y_gen = tf.ones((shape[0], 1))
                x_output = img

                gan_res = model_gan.train_on_batch(x_gen, [y_gen, x_output])
                model_dis.trainable = True

                pbar.set_description("Epoch : %d Dis loss: %f Gan total: %f Gan loss: %f Gan L1: %f P_ACC: %f ACC: %f" % (epoch, dis_res,
                                        gan_res[0], gan_res[1], gan_res[2], gan_res[5], gan_res[6]))

            # if epoch % 5 == 0:
            model_gen.save_weights(WEIGHTS_GEN + str(SCALE_STEP[steps]) + '_'+ str(epoch) + '.h5', overwrite=True)
            model_dis.save_weights(WEIGHTS_DIS + str(SCALE_STEP[steps]) + '_'+ str(epoch) + '.h5', overwrite=True)
            model_gan.save_weights(WEIGHTS_GAN + str(SCALE_STEP[steps]) + '_'+ str(epoch) + '.h5', overwrite=True)

            os.makedirs(DEMO_OUTPUT + str(SCALE_STEP[steps]) + '/'+ str(epoch), exist_ok=True)

            # validation
            for img in demo_test:
                img = tf.image.resize_with_pad(img, IMAGE_SHAPE[0], IMAGE_SHAPE[1])
                img = tf.cast(img, tf.float32)
                img /= 255.

                r_channel = img[:, :, :, 0]

                pred_gb = model_gen.predict(r_channel)

                for i in range(len(pred_gb)):
                    batch_g = pred_gb[i][:, :, 0]
                    batch_b = pred_gb[i][:, :, 1]
                    
                    batch_a = tf.expand_dims(batch_a, -1)
                    batch_b = tf.expand_dims(batch_b, -1)

                    pred_rgb = tf.concat([r_channel, batch_a, batch_b], axis=-1)
                    
                    pred_rgb *= 255.

                    pred_rgb = tf.cast(pred_rgb, tf.uint8)
                    plt.imshow(pred_rgb)

                    plt.savefig(DEMO_OUTPUT + str(SCALE_STEP[steps]) + '/'+ str(epoch) + '/'+ str(index)+'.png', dpi=200)
                    index +=1