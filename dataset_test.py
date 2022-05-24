from re import T
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
    batch_size = 1
    demo = False
    celebA_hq = tfds.load('CustomCelebahq',
                           data_dir=DATASET_DIR, split='train', shuffle_files=True)

    # celebA = tfds.load('CustomCeleba',
    #                        data_dir=DATASET_DIR, split='train', shuffle_files=True)
    #
    # train_data = celebA_hq.concatenate(celebA)
    train_data = celebA_hq
    train_data = train_data.padded_batch(1)
    train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)


    if demo:
    # prepare validation dataset
        filenames = os.listdir('./demo_images')
        filenames.sort()
        train_data = tf.data.Dataset.list_files('./demo_images/' + '*', shuffle=False)
        train_data = train_data.map(demo_prepare)
        train_data = train_data.batch(batch_size)


    for batch in train_data.take(100):
        if demo is False:
            batch = batch['image']

        img = tf.image.resize(batch, (512, 512))
        
        original = img
        
        img /= 255. # normalize image 0 ~ 1.
        img = tf.cast(img, tf.float32)
        
        lab = color.rgb2lab(img)
        
        
        
        
        l_channel = lab[:, :, :, :1]
        a_channel = lab[:, :, :, 1]
        b_channel = lab[:, :, :, 2]
        
        print('L max :', np.max(l_channel), 'L min :', np.min(l_channel))
        print('a max :', np.max(a_channel), 'a_min :', np.min(a_channel))
        print('b max :', np.max(b_channel), 'b_min :', np.min(b_channel))
        ab_channel = lab[:, :, :, 1:]

        l_channel = (l_channel - 50) / 50.
        ab_channel /= 127.
        
        
        norm_lab = tf.concat([l_channel , ab_channel], axis=-1)
        norm_lab_2 = norm_lab
        
        ssim_loss = tf.image.ssim(norm_lab, norm_lab_2, max_val=2.0)
        ms_ssim_loss = tf.image.ssim_multiscale(norm_lab, norm_lab_2, max_val=2.0)
        
        
        ssim_loss = tf.reduce_mean((1-ms_ssim_loss), axis=0)        
        print(ssim_loss)
            
        rgb = color.lab2rgb(lab)

        img = tf.cast(img, tf.float32) # if use ycbcr
        for i in range(batch_size):
            
            NORM_RGB = (img[i] / 127.5) - 1
            norm_lab = lab [i]
            R = norm_lab[:, :, 0]
            G = norm_lab[:, :, 1]
            B = norm_lab[:, :, 2]

            rows = 1
            cols = 3
            fig = plt.figure()

            ax0 = fig.add_subplot(rows, cols, 1)
            ax0.imshow(R)
            ax0.set_title('L channel')
            ax0.axis("off")

            ax0 = fig.add_subplot(rows, cols, 2)
            ax0.imshow(G)
            ax0.set_title('a channel')
            ax0.axis("off")

            ax0 = fig.add_subplot(rows, cols, 3)
            ax0.imshow(B)
            ax0.set_title('b channel')
            ax0.axis("off")

            # ax0 = fig.add_subplot(rows, cols, 4)
            # ax0.imshow(B)
            # ax0.set_title('B channel')
            # ax0.axis("off")

            plt.savefig('lab.png', dpi=500)
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



