import tensorflow as tf
import matplotlib.pyplot as plt
from utils.datasets import Dataset
import argparse
import tensorflow_io as tfio

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, help="데이터셋 다운로드 디렉토리 설정", default='./datasets/')

args = parser.parse_args()

DATASET_DIR = args.dataset_dir
IMAGE_SIZE = (512, 512)

# train_data = train_dataset_config.get_trainData(train_dataset_config.train_data)

if __name__ == "__main__":
    train_dataset_config = Dataset(DATASET_DIR, IMAGE_SIZE, batch_size=1, mode='train')
    train_data = train_dataset_config.dataset_test(train_dataset_config.train_data)



    for l, ab in train_data.take(100):


        L = l[0]
        a = ab[0][:, :, 0]
        b = ab[0][:, :, 1]


        rows = 1
        cols = 3

        fig = plt.figure()

        ax1 = fig.add_subplot(rows, cols, 1)
        ax1.imshow(L)
        ax1.set_title('L image')
        ax1.axis("off")

        ax2 = fig.add_subplot(rows, cols, 2)
        ax2.imshow(a)
        ax2.set_title('a image')
        ax2.axis("off")

        ax3 = fig.add_subplot(rows, cols, 3)
        ax3.imshow(b)
        ax3.set_title('b image')
        ax3.axis("off")

        plt.show()



