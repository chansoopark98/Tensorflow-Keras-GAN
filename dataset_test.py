import tensorflow as tf
import matplotlib.pyplot as plt
from utils.datasets import Dataset
import argparse
import tensorflow_io as tfio
from skimage import color

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, help="데이터셋 다운로드 디렉토리 설정", default='./datasets/')

args = parser.parse_args()

DATASET_DIR = args.dataset_dir
IMAGE_SIZE = (512, 512)

# train_data = train_dataset_config.get_trainData(train_dataset_config.train_data)

if __name__ == "__main__":
    train_dataset_config = Dataset(DATASET_DIR, IMAGE_SIZE, batch_size=1, mode='train', dataset='CustomCelebahq')
    # train_dataset_config = Dataset(DATASET_DIR, IMAGE_SIZE, batch_size=1, mode='train', dataset='CustomCeleba')
    train_data = train_dataset_config.dataset_test(train_dataset_config.train_data)



    for img in train_data.take(100):

        img = img[0]
        # img = tf.image.resize(img, (512, 512))
        img = tf.image.resize_with_pad(img, 256, 256)
        img = tf.cast(img, tf.float32) # if use ycbcr
        img /= 255. #TODO! if use lab

        lab = tfio.experimental.color.rgb_to_lab(img)
        l_cent = 50.
        l_norm = 100.
        ab_norm = 110.
        #
        """    
        def normalize_l(self, in_l):
            return (in_l - self.l_cent) / self.l_norm
    
    
        def unnormalize_l(self, in_l):
            return in_l * self.l_norm + self.l_cent
    
    
        def normalize_ab(self, in_ab):
            return in_ab / self.ab_norm
    
    
        def unnormalize_ab(self, in_ab):
            return in_ab * self.ab_norm
        """

        l = lab[:, :, 0]
        l = l.numpy()
        l = (l - l_cent) / l_norm
        l = l * l_norm + l_cent

        a = lab[:, :, 1]
        a = a.numpy()
        a = a / ab_norm
        a = a * ab_norm

        b = lab[:, :, 2]
        b = b.numpy()
        b = b / ab_norm
        b = b * ab_norm

        l = tf.expand_dims(l, axis=-1)
        a = tf.expand_dims(a, axis=-1)
        b = tf.expand_dims(b, axis=-1)

        lab = tf.concat([l, a, b], axis=-1)

        lab = tfio.experimental.color.lab_to_rgb(lab)

        rows = 1
        cols = 2
        fig = plt.figure()

        ax0 = fig.add_subplot(rows, cols, 1)
        ax0.imshow(lab)
        ax0.set_title('rgb->lab->rgb')
        ax0.axis("off")

        ax0 = fig.add_subplot(rows, cols, 2)
        ax0.imshow(img)
        ax0.set_title('original')
        ax0.axis("off")


        plt.show()



