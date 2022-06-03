import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_datasets as tfds
from tqdm import tqdm

def get_histogram():
    BATCH_SIZE = 8
    DATASET_DIR ='./datasets'
    test_data = tfds.load('CustomCelebahq',
                        data_dir=DATASET_DIR, split='train', shuffle_files=True)

    train_data = train_data.padded_batch(BATCH_SIZE)


if __name__ == '__main__':
    get_histogram()