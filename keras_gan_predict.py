from model.model_builder import build_dis, build_gen
import tensorflow as tf
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy, mean_absolute_error
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import tensorflow.keras.backend as K
from tqdm import tqdm
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import os

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
    optimizer = Adam(lr=lr, beta_1=momentum)
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

    model_dis.trainable = True
    model_dis.compile(loss=binary_crossentropy, optimizer=optimizer)

    return model_gen, model_dis, model_gan

if __name__ == '__main__':
    EPOCHS = 200
    BATCH_SIZE = 1
    LEARNING_RATE = 0.0005
    MOMENTUM = 0.5
    LAMBDA1 = 1
    LAMBDA2 = 100
    INPUT_SHAPE_GEN = (256, 256, 1)
    INPUT_SHAPE_DIS = (256, 256, 3)
    GEN_OUTPUT_CHANNEL = 2
    DATASET_DIR ='./datasets'
    WEIGHTS_GEN = './checkpoints/YUV_GAN_Gen.h5'
    WEIGHTS_DIS = './checkpoints/YUV_GAN_Dis.h5'
    WEIGHTS_GAN = './checkpoints/YUV_GAN_Gan.h5'

    model_gen, model_dis, model_gan = create_models(
        input_shape_gen=INPUT_SHAPE_GEN,
        input_shape_dis=INPUT_SHAPE_DIS,
        output_channels=GEN_OUTPUT_CHANNEL,
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        loss_weights=[LAMBDA1, LAMBDA2])

    model_gen.load_weights('./checkpoints/YUV_GAN_Gen.h5', by_name=True)
    train_data = tfds.load('CustomCelebahq',
                                data_dir=DATASET_DIR, split='train[:25%]', shuffle_files=True)
    number_train = train_data.reduce(0, lambda x, _: x + 1).numpy()
    print("학습 데이터 개수", number_train)
    steps_per_epoch = number_train // BATCH_SIZE
    train_data = train_data.shuffle(1024)
    train_data = train_data.padded_batch(BATCH_SIZE)
    # train_data = train_data.repeat(EPOCHS)
    train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)

    save_path = './checkpoints/results/' + 'gan' + '/'
    os.makedirs(save_path, exist_ok=True)
    batch_index = 0
    for epoch in range(EPOCHS):
        pbar = tqdm(train_data, total=steps_per_epoch, desc='Batch', leave=True, disable=False)

        for features in pbar:
            img = tf.cast(features['image'], tf.uint8)
            shape = img.shape
            img = tf.image.resize(img, (INPUT_SHAPE_GEN[0], INPUT_SHAPE_GEN[1]), tf.image.ResizeMethod.BILINEAR)
            img /= 255
            img = tf.cast(img, tf.float32)
            yuv = tf.image.rgb_to_yuv(img)

            y = yuv[:, :, :, 0]
            y = tf.cast(y, tf.float32)
            y *= 255.
            # y = (y / 127.5) - 1.0
            y /= 255.
            y = tf.expand_dims(y, axis=-1)

            u = yuv[:, :, :, 1]
            u = tf.cast(u, tf.float32)
            u = (u + 0.5) * 255.
            # u = (u / 127.5) - 1.0
            u /= 255.
            u = tf.expand_dims(u, axis=-1)

            v = yuv[:, :, :, 2]
            v = tf.cast(v, tf.float32)
            v = (v + 0.5) * 255.
            # v = (v / 127.5) - 1.0
            v /= 255.
            v = tf.expand_dims(v, axis=-1)

            uv = tf.concat([u, v], axis=-1)

            pred_yuv = model_gen.predict(y)

            y = y[0]
            u =u[0]
            v =v[0]
            pred_u = pred_yuv[0][:, :, 0]
            pred_v = pred_yuv[0][:, :, 1]

            pred_u *= 255.
            pred_u = (pred_u / 255.) - 0.5

            pred_v *= 255.
            pred_v = (pred_v / 255.) - 0.5

            pred_u = tf.expand_dims(pred_u, -1)
            pred_v = tf.expand_dims(pred_v, -1)

            pred_yuv = tf.concat([y, pred_u, pred_v], axis=-1)

            pred_yuv = tf.image.yuv_to_rgb(pred_yuv)



            u *= 255.
            u = (u / 255.) - 0.5

            v *= 255.
            v = (v / 255.) - 0.5



            gt_yuv = tf.concat([y, u, v], axis=-1)
            gt_yuv = tf.image.yuv_to_rgb(gt_yuv)

            rows = 1
            cols = 2
            fig = plt.figure()

            ax0 = fig.add_subplot(rows, cols, 1)
            ax0.imshow(pred_yuv)
            ax0.set_title('Prediction')
            ax0.axis("off")

            ax1 = fig.add_subplot(rows, cols, 2)
            ax1.imshow(gt_yuv)
            ax1.set_title('Groundtruth')
            ax1.axis("off")

            plt.savefig(save_path + str(batch_index) + 'output.png', dpi=300)
            # pred = tf.cast(pred, tf.int32)
            # plt.show()
            # tf.keras.preprocessing.image.save_img(save_path + str(batch_index) + '_1_input.jpg', output)
            # tf.keras.preprocessing.image.save_img(save_path + str(batch_index) + '_2_gt.jpg', img[0])
            # tf.keras.preprocessing.image.save_img(save_path + str(batch_index) + '_3_out.jpg', pred)

            batch_index += 1
