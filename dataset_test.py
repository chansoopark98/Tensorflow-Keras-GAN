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



    for Y, U, V in train_data.take(100):
        y = Y[0].numpy()
        u = U[0].numpy()
        v = V[0].numpy()

        y = (y + 1) * 127.5
        u = (u + 1) * 127.5
        v = (v + 1) * 127.5

        yuv = tf.concat([y, u, v], axis=-1)
        yuv /= 255.
        img = tf.image.yuv_to_rgb(yuv)

        img_255 = img * 255
        img_255 = tf.cast(img_255, tf.uint8)
        rows = 1
        cols = 2
        fig = plt.figure()

        ax0 = fig.add_subplot(rows, cols, 1)
        ax0.imshow(img)
        ax0.set_title('0 to 1')
        ax0.axis("off")

        ax0 = fig.add_subplot(rows, cols, 2)
        ax0.imshow(img_255)
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



