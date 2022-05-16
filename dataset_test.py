import tensorflow as tf
import matplotlib.pyplot as plt
from utils.datasets import Dataset
import argparse
import tensorflow_io as tfio
from skimage import color
import tensorflow_datasets as tfds
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, help="데이터셋 다운로드 디렉토리 설정", default='./datasets/')

args = parser.parse_args()

DATASET_DIR = args.dataset_dir
IMAGE_SIZE = (512, 512)

def demo_prepare(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    return (img)
# train_data = train_dataset_config.get_trainData(train_dataset_config.train_data)

if __name__ == "__main__":

    celebA_hq = tfds.load('CustomCelebahq',
                           data_dir=DATASET_DIR, split='train', shuffle_files=True)

    # celebA = tfds.load('CustomCeleba',
    #                        data_dir=DATASET_DIR, split='train', shuffle_files=True)
    #
    # train_data = celebA_hq.concatenate(celebA)
    train_data = celebA_hq
    train_data = train_data.padded_batch(1)
    train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)

    
    # prepare validation dataset
    filenames = os.listdir('./demo_images')
    filenames.sort()
    train_data = tf.data.Dataset.list_files('./demo_images/' + '*', shuffle=False)
    train_data = train_data.map(demo_prepare)
    train_data = train_data.batch(1)


    for batch in train_data.take(5):

        # img = batch['image']
        img = batch[0]
        img = tf.image.resize(img, (512, 512))
        
        original = img

        # gray = tf.image.rgb_to_grayscale(img)
        gray = color.rgb2gray(img)
        
        img = tf.cast(img, tf.float32) # if use ycbcr
        NORM_RGB = (img / 127.5) - 1
        
        R = NORM_RGB[:, :, :1]
        GB = NORM_RGB[:, :, 1:]
        
        

        rows = 1
        cols = 4
        fig = plt.figure()

        ax0 = fig.add_subplot(rows, cols, 1)
        ax0.imshow(R[:, :, 0])
        ax0.set_title('r')
        ax0.axis("off")

        ax0 = fig.add_subplot(rows, cols, 2)
        ax0.imshow(GB[:, :, 0])
        ax0.set_title('g')
        ax0.axis("off")

        ax0 = fig.add_subplot(rows, cols, 3)
        ax0.imshow(GB[:, :, 1])
        ax0.set_title('b')
        ax0.axis("off")

        ax0 = fig.add_subplot(rows, cols, 4)
        ax0.imshow(gray)
        ax0.set_title('b')
        ax0.axis("off")


        plt.show()


        # yuv = tfio.experimental.color.rgb_to_yuv(lab)
        
        # l_cent = 50.
        # l_norm = 100.
        # ab_norm = 110.

        # l = lab[:, :, 0]
        # l = l.numpy()
        # l = (l - l_cent) / l_norm
        # l = l * l_norm + l_cent

        # a = lab[:, :, 1]
        # a = a.numpy()
        # a = a / ab_norm
        # a = a * ab_norm

        # b = lab[:, :, 2]
        # b = b.numpy()
        # b = b / ab_norm
        # b = b * ab_norm

        # l = tf.expand_dims(l, axis=-1)
        # a = tf.expand_dims(a, axis=-1)
        # b = tf.expand_dims(b, axis=-1)

        # lab = tf.concat([l, a, b], axis=-1)

        # lab = tfio.experimental.color.lab_to_rgb(lab)

        # rows = 1
        # cols = 2
        # fig = plt.figure()

        # ax0 = fig.add_subplot(rows, cols, 1)
        # ax0.imshow(lab)
        # ax0.set_title('rgb->lab->rgb')
        # ax0.axis("off")

        # ax0 = fig.add_subplot(rows, cols, 2)
        # ax0.imshow(img)
        # ax0.set_title('original')
        # ax0.axis("off")


        # plt.show()



