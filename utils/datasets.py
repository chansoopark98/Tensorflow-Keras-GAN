import tensorflow_io as tfio
import tensorflow_datasets as tfds
import tensorflow as tf
from skimage import color
AUTO = tf.data.experimental.AUTOTUNE


class Dataset:
    def __init__(self, data_dir, image_size, batch_size, mode, dataset='CustomCelebahq'):
        """from i
        Args:
            data_dir: 데이터셋 상대 경로 ( default : './datasets/' )
            image_size: 백본에 따른 이미지 해상도 크기
            batch_size: 배치 사이즈 크기
            dataset: 데이터셋 종류 (celebA: 'CustomCeleba', celebAHQ: 'CustomCelebahq')
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.dataset_name = dataset

        self.l_cent = 50.
        self.l_norm = 100.
        self.ab_norm = 110.

        if mode == 'train':
            self.train_data, self.number_train = self._load_train_datasets()
        elif mode == 'validation':
            self.valid_data, self.number_valid = self._load_valid_datasets()

    def _load_valid_datasets(self):
        valid_data = tfds.load(self.dataset_name,
                               data_dir=self.data_dir, split='train[:1%]')

        number_valid = valid_data.reduce(0, lambda x, _: x + 1).numpy()
        print("검증 데이터 개수:", number_valid)

        return valid_data, number_valid

    def _load_train_datasets(self):
        train_data = tfds.load(self.dataset_name,
                               data_dir=self.data_dir, split='train[1%:]')

        # low_quality_train_data = tfds.load('CustomCeleba',
        #                        data_dir=self.data_dir, split='train')
        #
        # train_data = train_data.concatenate(low_quality_train_data)

        number_train = train_data.reduce(0, lambda x, _: x + 1).numpy()
        print("학습 데이터 개수", number_train)

        return train_data, number_train

    @tf.function
    def zoom(self, x, scale_min=0.8, scale_max=1.1):
        h, w, = 256, 256
        scale = tf.random.uniform([], scale_min, scale_max)
        nh = h * scale
        nw = w * scale
        x = tf.image.resize(x, (nh, nw), method=tf.image.ResizeMethod.BILINEAR)
        x = tf.image.resize_with_crop_or_pad(x, h, w)
        return x

    def load_test(self, sample):
        img = sample['image']

        img = tf.image.resize(img, (self.image_size[0], self.image_size[1]), tf.image.ResizeMethod.BILINEAR)
        gray = tfio.experimental.color.rgb_to_grayscale(img)
        gray = tf.cast(gray, tf.float32)
        gray /= 127.5
        gray -= 1.

        img /= 255.

        lab = tfio.experimental.color.rgb_to_lab(img)

        l = lab[:, :, 0]
        l = (l - self.l_cent) / self.l_norm

        a = lab[:, :, 1]
        a = a / self.ab_norm

        b = lab[:, :, 2]
        b = b / self.ab_norm

        l = tf.expand_dims(l, axis=-1)
        a = tf.expand_dims(a, axis=-1)
        b = tf.expand_dims(b, axis=-1)

        lab = tf.concat([l, a, b], axis=-1)

        return (gray, lab)

    @tf.function
    def preprocess(self, sample):
        img = sample['image']
        img = tf.cast(img, tf.float32)

        # data augmentation
        if tf.random.uniform([], minval=0, maxval=1) > 0.5:
            img = tf.image.flip_left_right(img)
        # if tf.random.uniform([], minval=0, maxval=1) > 0.5:
        #     img = tf.image.random_brightness(img, 0.2)
        # if tf.random.uniform([], minval=0, maxval=1) > 0.5:
        #     img = self.zoom(img)


        img = tf.image.resize(img, (self.image_size[0], self.image_size[1]), tf.image.ResizeMethod.BILINEAR)
        gray = tfio.experimental.color.rgb_to_grayscale(img)
        gray = tf.cast(gray, tf.float32)
        gray /= 127.5
        gray -= 1.


        img /= 255.

        lab = tfio.experimental.color.rgb_to_lab(img)

        l = lab[:, :, 0]
        l = (l - self.l_cent) / self.l_norm

        a = lab[:, :, 1]
        a = a / self.ab_norm

        b = lab[:, :, 2]
        b = b / self.ab_norm

        l = tf.expand_dims(l, axis=-1)
        a = tf.expand_dims(a, axis=-1)
        b = tf.expand_dims(b, axis=-1)

        lab = tf.concat([l, a, b], axis=-1)

        return (gray, lab)

    @tf.function
    def preprocess_valid(self, sample):
        img = sample['image']

        img = tf.image.resize(img, (self.image_size[0], self.image_size[1]), tf.image.ResizeMethod.BILINEAR)
        gray = tfio.experimental.color.rgb_to_grayscale(img)
        gray = tf.cast(gray, tf.float32)
        gray /= 127.5
        gray -= 1.

        img /= 255.

        lab = tfio.experimental.color.rgb_to_lab(img)

        l = lab[:, :, 0]
        l = (l - self.l_cent) / self.l_norm

        a = lab[:, :, 1]
        a = a / self.ab_norm

        b = lab[:, :, 2]
        b = b / self.ab_norm

        l = tf.expand_dims(l, axis=-1)
        a = tf.expand_dims(a, axis=-1)
        b = tf.expand_dims(b, axis=-1)

        lab = tf.concat([l, a, b], axis=-1)

        return (gray, lab)

    @tf.function
    def load_original_img(self, sample):
        img = sample['image']

        return (img)


    def get_trainData(self, train_data):
        train_data = train_data.shuffle(1024)
        train_data = train_data.map(self.preprocess, num_parallel_calls=AUTO)
        train_data = train_data.padded_batch(self.batch_size)
        train_data = train_data.prefetch(AUTO)
        train_data = train_data.repeat()

        return train_data

    def get_validData(self, valid_data):
        valid_data = valid_data.map(self.preprocess_valid, num_parallel_calls=AUTO)
        valid_data = valid_data.padded_batch(self.batch_size).prefetch(AUTO)
        return valid_data

    def get_testData(self, valid_data):
        valid_data = valid_data.map(self.load_test)
        valid_data = valid_data.batch(self.batch_size).prefetch(AUTO)
        return valid_data

    def dataset_test(self, train_data):
        train_data = train_data.map(self.load_original_img)
        train_data = train_data.batch(self.batch_size).prefetch(AUTO)
        return train_data