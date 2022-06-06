import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_datasets as tfds
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

def get_histogram():
    BATCH_SIZE = 8
    DATASET_DIR ='./datasets'
    train_data = tfds.load('CustomCelebahq',
                        data_dir=DATASET_DIR, split='train', shuffle_files=True)

    train_data = train_data.padded_batch(BATCH_SIZE)
    number_train = train_data.reduce(0, lambda x, _: x + 1).numpy()
    print("학습 데이터 개수", number_train)
    steps_per_epoch = number_train // BATCH_SIZE

    pbar = tqdm(train_data, total=steps_per_epoch, desc='Batch', leave=True, disable=False)

    
    for sample in pbar:
        img = sample['image']
        img /= 255
        img = tf.cast(img, tf.float32)
        lab = tfio.experimental.color.rgb_to_lab(img).numpy()

        l_channel = lab[:, :, :, 0]
        a_channel = lab[:, :, :, 1] 
        a_channel += 127
        b_channel = lab[:, :, :, 2]
        b_channel += 127

        rows = 1
        cols = 3
        fig = plt.figure()
        

        ax0 = fig.add_subplot(rows, cols, 1)
        ax0.hist(l_channel.ravel(), 100, [0, 100])
        ax0.set_title('L channel')
        ax0.axis("off")

        ax1 = fig.add_subplot(rows, cols, 2)
        ax1.hist(a_channel.ravel(), 256, [0, 256])
        ax1.set_title('a channel')
        ax1.axis("off")

        ax2 = fig.add_subplot(rows, cols, 3)
        ax2.hist(b_channel.ravel(), 256, [0, 256])
        ax2.set_title('b channel')
        ax2.axis("off")
            
        plt.show()


if __name__ == '__main__':
    get_histogram()