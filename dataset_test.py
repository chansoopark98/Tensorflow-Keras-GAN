import tensorflow as tf
import matplotlib.pyplot as plt
from utils.datasets import Dataset
import argparse
import tensorflow_io as tfio
from skimage.color import rgb2lab, rgb2gray

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, help="데이터셋 다운로드 디렉토리 설정", default='./datasets/')

args = parser.parse_args()

DATASET_DIR = args.dataset_dir
IMAGE_SIZE = (512, 512)

# train_data = train_dataset_config.get_trainData(train_dataset_config.train_data)

if __name__ == "__main__":
    train_dataset_config = Dataset(DATASET_DIR, IMAGE_SIZE, batch_size=1, mode='train')
    train_data = train_dataset_config.dataset_test(train_dataset_config.train_data)



    for data in train_data.take(100):


        """
        ### data norm
            normalized_input = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
            return 2*normalized_input - 1
        """
        img = data[0].numpy()

        # Generate L,a,b channels image From input RGB data.
        img = tf.cast(img, tf.uint8)
        # img /= 255. # input is Float type
        """ for Lab color"""
        # img_lab = tfio.experimental.color.rgb_to_lab(img)
        # img_lab = (img_lab + 128) / 255

        # L = img_lab[:, :, 0]
        # L = (L / 50.) - 1.

        # a = img_lab[:, :, 1]
        # a = ((a+127.)/255.) * 2 - 1.

        # b = img_lab[:, :, 2]
        # b = ((b + 127.) / 255.) * 2 - 1.

        # L = tf.expand_dims(L, -1)
        # a = tf.expand_dims(a, -1)
        # b = tf.expand_dims(b, -1)

        # ab_channel = tf.concat([a, b], axis=-1)


        img_YCbCr = tfio.experimental.color.rgb_to_ycbcr(img)
        img_grayscale = tfio.experimental.color.rgb_to_grayscale(img)
        Y = img_YCbCr[:, :, 0]
        Y = tf.expand_dims(Y, axis=-1)

        Gray = img_grayscale
        Gray_3channel = tf.concat([Gray, Gray, Gray], axis=-1)
        gray_ycbcr = tfio.experimental.color.rgb_to_ycbcr(Gray_3channel)
        gray_Y = gray_ycbcr[:, :, 0]
        gray_Y = tf.expand_dims(gray_Y, axis=-1)

        Cb = img_YCbCr[:, :, 1]
        Cb = tf.expand_dims(Cb, axis=-1)
        Cr = img_YCbCr[:, :, 2]
        Cr = tf.expand_dims(Cr, axis=-1)

        convert_YCbCR = tf.concat([gray_Y, Cb, Cr], axis=-1)
        convert_YCbCR = tf.cast(convert_YCbCR, tf.uint8)
        convert_RGB = tfio.experimental.color.ycbcr_to_rgb(convert_YCbCR)
        rows = 1
        cols = 4



        fig = plt.figure()

        ax0 = fig.add_subplot(rows, cols, 1)
        ax0.imshow(convert_RGB)
        ax0.set_title('RGB image')
        ax0.axis("off")

        ax1 = fig.add_subplot(rows, cols, 2)
        ax1.imshow(Y)
        ax1.set_title('Y image')
        ax1.axis("off")

        ax2 = fig.add_subplot(rows, cols, 3)
        ax2.imshow(Cb)
        ax2.set_title('Cb image')
        ax2.axis("off")

        ax3 = fig.add_subplot(rows, cols, 4)
        ax3.imshow(Cr)
        ax3.set_title('Cr image')
        ax3.axis("off")

        plt.show()



