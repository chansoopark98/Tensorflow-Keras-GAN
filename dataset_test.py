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



    for data in train_data.take(100):


        """
        ### data norm
            normalized_input = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
            return 2*normalized_input - 1
        """
        img = data[0].numpy()

        # Generate L,a,b channels image From input RGB data.
        img /= 255. # input is Float type

        img_lab = tfio.experimental.color.rgb_to_lab(img)
        L = img_lab[:, :, 0]
        L = (L / 50.) - 1.

        a = img_lab[:, :, 1]
        a = ((a+127.)/255.) * 2 - 1.

        b = img_lab[:, :, 2]
        b = ((b + 127.) / 255.) * 2 - 1.

        L = tf.expand_dims(L, -1)
        a = tf.expand_dims(a, -1)
        b = tf.expand_dims(b, -1)
        #
        ab_channel = tf.concat([a, b], axis=-1)

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



