import tensorflow as tf
import matplotlib.pyplot as plt
from utils.datasets import Dataset
import argparse
import tensorflow_io as tfio
from skimage import color

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, help="데이터셋 다운로드 디렉토리 설정", default='./datasets/')

args = parser.parse_args()

DATASET_DIR = args.dataset_dir
IMAGE_SIZE = (512, 512)

# train_data = train_dataset_config.get_trainData(train_dataset_config.train_data)

if __name__ == "__main__":
    train_dataset_config = Dataset(DATASET_DIR, IMAGE_SIZE, batch_size=1, mode='train', dataset='CustomCelebahq')
    # train_dataset_config = Dataset(DATASET_DIR, IMAGE_SIZE, batch_size=1, mode='train', dataset='CustomCeleba')
    train_data = train_dataset_config.dataset_test(train_dataset_config.train_data)



    for img in train_data.take(100):
        img = tf.cast(img, tf.uint8)
        img = img[0]
        # img = tf.image.resize(img, (512, 512))
        img = tf.image.resize_with_pad(img, 256, 256)
        img /= 255.

        # lab = tfio.experimental.color.rgb_to_lab(img)
        ten_lab = tfio.experimental.color.rgb_to_lab(img)


        l = ten_lab[:, :, 0] # normalize 0 ~ 100 to -1 ~ 1
        l = l.numpy()
        l = (l - 50) / 50.
        l = (l * 50) + 50


        a = ten_lab[:, :, 1]
        a = a.numpy()
        a /= 127.5
        a *= 127.5

        b = ten_lab[:, :, 2]
        b /= 127.5
        b *= 127.5

        l = tf.expand_dims(l, axis=-1)
        a = tf.expand_dims(a, axis=-1)
        b = tf.expand_dims(b, axis=-1)

        lab = tf.concat([l, a, b], axis=-1)

        lab = tfio.experimental.color.lab_to_rgb(lab)

        rows = 1
        cols = 2
        fig = plt.figure()

        ax0 = fig.add_subplot(rows, cols, 1)
        ax0.imshow(lab)
        ax0.set_title('rgb->lab->rgb')
        ax0.axis("off")

        ax0 = fig.add_subplot(rows, cols, 2)
        ax0.imshow(img)
        ax0.set_title('original')
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



