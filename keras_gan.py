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
    dis_out = discriminator(concatenate([gen_out, input], axis=3))

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

def demo_prepare(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)

    return (img)


if __name__ == '__main__':
    EPOCHS = 100
    BATCH_SIZE = 32
    # LEARNING_RATE = 0.0005
    LEARNING_RATE = 0.0002
    MOMENTUM = 0.5
    LAMBDA1 = 1
    LAMBDA2 = 100
    INPUT_SHAPE_GEN = (256, 256, 1)
    INPUT_SHAPE_DIS = (256, 256, 3)
    GEN_OUTPUT_CHANNEL = 2
    DATASET_DIR ='./datasets'
    # WEIGHTS_GEN = './checkpoints/YUV_GAN_Gen.h5'
    WEIGHTS_GEN = './checkpoints/YUV_GAN_Gen_'
    # WEIGHTS_DIS = './checkpoints/YUV_GAN_Dis.h5'
    WEIGHTS_DIS = './checkpoints/YUV_GAN_Dis_'
    # WEIGHTS_GAN = './checkpoints/YUV_GAN_Gan.h5'
    WEIGHTS_GAN = './checkpoints/YUV_GAN_Gan_'

    model_gen, model_dis, model_gan = create_models(
        input_shape_gen=INPUT_SHAPE_GEN,
        input_shape_dis=INPUT_SHAPE_DIS,
        output_channels=GEN_OUTPUT_CHANNEL,
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        loss_weights=[LAMBDA1, LAMBDA2])

    # model_gen.load_weights(WEIGHTS_GEN + '0.h5')
    # model_dis.load_weights(WEIGHTS_DIS + '0.h5')
    # model_gan.load_weights(WEIGHTS_GAN + '0.h5')

    train_data = tfds.load('CustomCelebahq',
                           data_dir=DATASET_DIR, split='train', shuffle_files=True)

    number_train = train_data.reduce(0, lambda x, _: x + 1).numpy()
    print("학습 데이터 개수", number_train)
    steps_per_epoch = number_train // BATCH_SIZE
    train_data = train_data.shuffle(1024)
    train_data = train_data.padded_batch(BATCH_SIZE)
    # train_data = train_data.repeat(EPOCHS)
    train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)

    # prepare validation dataset
    filenames = os.listdir('./demo_images')
    filenames.sort()
    demo_imgs = tf.data.Dataset.list_files('./demo_images/' + '*', shuffle=False)
    demo_test = demo_imgs.map(demo_prepare)
    demo_test = demo_test.batch(1)
    demo_steps = len(filenames) // 1
    demo_path = './demo_outputs/' + 'demo/'
    os.makedirs(demo_path, exist_ok=True)


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
            img = tf.cast(features['image'], tf.uint8)
            shape = img.shape

            img = tf.cast(img, tf.float32)
            img = tf.image.resize(img, (INPUT_SHAPE_GEN[0], INPUT_SHAPE_GEN[1]), tf.image.ResizeMethod.BILINEAR)

            # data augmentation
            if tf.random.uniform([], minval=0, maxval=1) > 0.5:
                img = tf.image.flip_left_right(img)

            img /= 255.

            lab = tfio.experimental.color.rgb_to_lab(img)
            
            l = lab[:, :, :, 0]
            l = (l - 50) / 50.

            a = lab[:, :, :, 1]
            a /= 128.

            b = lab[:, :, :, 2]
            b /= 128.

            l = tf.expand_dims(l, axis=-1)
            a = tf.expand_dims(a, axis=-1)
            b = tf.expand_dims(b, axis=-1)

            ab = tf.concat([a, b], axis=-1)

            if batch_counter % 2 == 0:
                toggle = not toggle
                if toggle:
                    x_dis = tf.concat((model_gen.predict(l), l), axis=3)
                    y_dis = tf.zeros((shape[0], 1)) # TODO: np to tf
                else:
                    x_dis = tf.concat((ab, l), axis=3)
                    # y_dis = tf.ones((shape[0], 1))
                    # y_dis = tf.ones((shape[0], 1)) * 0.9
                    y_dis = tf.random.uniform(shape=[shape[0]], minval=0.9, maxval=1)
                    # y_dis = tf.random.uniform(shape[0], minval=0.9, maxval=1.0)
                    # y_dis = np.random.uniform(low=0.9, high=1, size=BATCH_SIZE)

                dis_res = model_dis.train_on_batch(x_dis, y_dis)

            model_dis.trainable = False
            x_gen = l
            y_gen = tf.ones((shape[0], 1))
            x_output = ab
            gan_res = model_gan.train_on_batch(x_gen, [y_gen, x_output])
            model_dis.trainable = True

            pbar.set_description("Epoch : %d Dis loss: %f Gan total: %f Gan loss: %f Gan L1: %f P_ACC: %f ACC: %f" % (epoch, dis_res,
                                    gan_res[0], gan_res[1], gan_res[2], gan_res[5], gan_res[6]))

        if epoch % 5 == 0:
            model_gen.save_weights(WEIGHTS_GEN + str(epoch) + '.h5', overwrite=True)
            model_dis.save_weights(WEIGHTS_DIS + str(epoch) + '.h5', overwrite=True)
            model_gan.save_weights(WEIGHTS_GAN + str(epoch) + '.h5', overwrite=True)

            os.makedirs(demo_path + str(epoch), exist_ok=True)

            # validation
            for img in demo_test:
                img = tf.image.resize_with_pad(img, INPUT_SHAPE_GEN[0], INPUT_SHAPE_GEN[1])
                img = tf.cast(img, tf.float32)
                img /= 255.

                lab = tfio.experimental.color.rgb_to_lab(img)

                l = lab[:, :, :, 0]
                l = (l - 50) / 50.

                pred_lab = model_gen.predict(l)

                for i in range(len(pred_lab)):
                    batch_a = pred_lab[i][:, :, 0]
                    batch_b = pred_lab[i][:, :, 1]

                    l = (l * 50) + 50
                    batch_a *= 128.
                    batch_b *= 128.

                    batch_l = tf.expand_dims(l[i], -1)
                    batch_a = tf.expand_dims(batch_a, -1)
                    batch_b = tf.expand_dims(batch_b, -1)

                    pred_lab = tf.concat([batch_l, batch_a, batch_b], axis=-1)

                    pred_lab = tfio.experimental.color.lab_to_rgb(pred_lab)

                    plt.imshow(pred_lab)

                    plt.savefig(demo_path + str(epoch) + '/'+ str(index)+'.png', dpi=300)
                    index +=1