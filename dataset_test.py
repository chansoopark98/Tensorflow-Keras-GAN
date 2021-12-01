import tensorflow as tf
import matplotlib.pyplot as plt
from utils.datasets import Dataset
import argparse
import tensorflow_io as tfio

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, help="데이터셋 다운로드 디렉토리 설정", default='./datasets/')

args = parser.parse_args()

DATASET_DIR = args.dataset_dir
IMAGE_SIZE = (224, 224)

# train_data = train_dataset_config.get_trainData(train_dataset_config.train_data)

if __name__ == "__main__":
    train_dataset_config = Dataset(DATASET_DIR, IMAGE_SIZE, batch_size=1, mode='train')
    train_data = train_dataset_config.get_trainData(train_dataset_config.train_data)


    buffer = ''
    id_list = []
    stack = 0
    batch_index = 0
    # save_path = './checkpoints/results/' + SAVE_MODEL_NAME + '/'

    for r, img in train_data.take(100):


        red = r[0]
        img = img[0]
        img = tf.cast(img, tf.uint8)

        plt.imshow(red)
        plt.show()

        plt.imshow(img)
        plt.show()


