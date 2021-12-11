import tensorflow as tf
import matplotlib.pyplot as plt
from utils.datasets import Dataset
import argparse
import tensorflow_io as tfio
from skimage.color import rgb2ycbcr, ycbcr2rgb

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, help="데이터셋 다운로드 디렉토리 설정", default='./datasets/')

args = parser.parse_args()

DATASET_DIR = args.dataset_dir
IMAGE_SIZE = (512, 512)

# train_data = train_dataset_config.get_trainData(train_dataset_config.train_data)

if __name__ == "__main__":
    train_dataset_config = Dataset(DATASET_DIR, IMAGE_SIZE, batch_size=1, mode='train')
    train_data = train_dataset_config.dataset_test(train_dataset_config.train_data)



    for img in train_data.take(100):
        img = tf.cast(img, tf.uint8)
        img = img[0]
        img = tf.image.resize(img, (512, 512))
        img /= 255
        img = tf.cast(img, tf.float32)
        yuv = tf.image.rgb_to_yuv(img)
        # yuv *= 255

        y = yuv[:, :, 0]
        y = tf.cast(y, tf.float32)
        y = y.numpy()
        y *= 255.
        y = (y / 127.5) - 1.0
        y = tf.expand_dims(y, axis=-1)

        u = yuv[:, :, 1]
        u = tf.cast(u, tf.float32)
        u = u.numpy()
        u = (u + 0.5) * 255.
        u = (u / 127.5) - 1.0
        u = tf.expand_dims(u, axis=-1)

        v = yuv[:, :, 2]
        v = tf.cast(v, tf.float32)
        v = v.numpy()
        v = (v + 0.5) * 255.
        v = (v / 127.5) - 1.0
        v = tf.expand_dims(v, axis=-1)

        # 복원
        y = (y + 1) * 127.5
        y = (y / 255.)

        u = (u + 1) * 127.5
        u = (u / 255.) - 0.5

        v = (v + 1) * 127.5
        v = (v / 255.) - 0.5

        yuv = tf.concat([y, u, v], axis=-1)
        # yuv /= 255.
        yuv = tf.image.yuv_to_rgb(yuv)
        # yuv = tf.cast(yuv, tf.int32)
        rows = 1
        cols = 2
        fig = plt.figure()

        ax0 = fig.add_subplot(rows, cols, 1)
        ax0.imshow(yuv)
        ax0.set_title('0 to 1')
        ax0.axis("off")

        ax0 = fig.add_subplot(rows, cols, 2)
        ax0.imshow(yuv)
        ax0.set_title('0 to 255')
        ax0.axis("off")

        # ax1 = fig.add_subplot(rows, cols, 2)
        # ax1.imshow(Gray)
        # ax1.set_title('Gray image')
        # ax1.axis("off")
        #
        # ax2 = fig.add_subplot(rows, cols, 3)
        # ax2.imshow(Cb)
        # ax2.set_title('Cb image')
        # ax2.axis("off")
        #
        # ax3 = fig.add_subplot(rows, cols, 4)
        # ax3.imshow(Cr)
        # ax3.set_title('Cr image')
        # ax3.axis("off")

        plt.show()



