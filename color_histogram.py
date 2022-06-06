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
        print('batch')
        img = sample['image'].numpy()
        
        plt.hist(img.ravel(), 256, [0, 256])
        plt.show()


if __name__ == '__main__':
    get_histogram()