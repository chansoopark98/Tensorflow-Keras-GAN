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
    batch_size = 8
    demo = False
    celebA_hq = tfds.load('CustomCelebahq',
                           data_dir=DATASET_DIR, split='train')

    # celebA = tfds.load('CustomCeleba',
    #                        data_dir=DATASET_DIR, split='train', shuffle_files=True)
    #
    # train_data = celebA_hq.concatenate(celebA)
    train_data = celebA_hq
    train_data = train_data.padded_batch(batch_size)
    train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)


    if demo:
    # prepare validation dataset
        filenames = os.listdir('./demo_images')
        filenames.sort()
        train_data = tf.data.Dataset.list_files('./demo_images/' + '*', shuffle=False)
        train_data = train_data.map(demo_prepare)
        train_data = train_data.batch(batch_size)

    idx = 1
    
    for batch in train_data.take(100):
        if demo is False:
            batch = batch['image']

        img = tf.image.resize(batch, (512, 512))        
        
        original = img
        
        img /= 255. # normalize image 0 ~ 1.
        img = tf.cast(img, tf.float32)
        
        lab = tfio.experimental.color.rgb_to_lab(img)
        ski_lab = color.rgb2lab(img)
        
        print(ski_lab - lab)

        
        for i in range(batch_size):    
            
            l_channel = lab[i, :, :, 0]
            a_channel = lab[i, :, :, 1]
            b_channel = lab[i, :, :, 2]
        
            rows = 1
            cols = 4
            fig = plt.figure()
            
            ax0 = fig.add_subplot(rows, cols, 1)
            ax0.imshow(img[i])
            ax0.set_title('RGB')
            ax0.axis("off")

            ax0 = fig.add_subplot(rows, cols, 2)
            ax0.imshow(l_channel)
            ax0.set_title('L channel')
            ax0.axis("off")

            ax0 = fig.add_subplot(rows, cols, 3)
            ax0.imshow(a_channel)
            ax0.set_title('a channel')
            ax0.axis("off")

            ax0 = fig.add_subplot(rows, cols, 4)
            ax0.imshow(b_channel)
            ax0.set_title('b channel')
            ax0.axis("off")
            
            

            # ax0 = fig.add_subplot(rows, cols, 4)
            # ax0.imshow(B)
            # ax0.set_title('B channel')
            # ax0.axis("off")


            plt.savefig('./test/' + str(idx)+'.png', dpi=300)
            idx += 1

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



