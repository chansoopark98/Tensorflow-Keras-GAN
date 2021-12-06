import tensorflow_io as tfio
import tensorflow_datasets as tfds
import tensorflow as tf
AUTO = tf.data.experimental.AUTOTUNE


class Dataset:
    def __init__(self, data_dir, image_size, batch_size, mode, dataset='CustomCelebahq'):
        """
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

        if mode == 'train':
            self.train_data, self.number_train = self._load_train_datasets()
        elif mode == 'validation':
            self.valid_data, self.number_valid = self._load_valid_datasets()

    def _load_valid_datasets(self):
        valid_data = tfds.load(self.dataset_name,
                               data_dir=self.data_dir, split='train[:5%]')

        number_valid = valid_data.reduce(0, lambda x, _: x + 1).numpy()
        print("검증 데이터 개수:", number_valid)

        return valid_data, number_valid

    def _load_train_datasets(self):
        train_data = tfds.load(self.dataset_name,
                               data_dir=self.data_dir, split='train[5%:]')

        number_train = train_data.reduce(0, lambda x, _: x + 1).numpy()
        print("학습 데이터 개수", number_train)

        return train_data, number_train


    def load_test(self, sample):
        img = tf.cast(sample['image'], tf.float32)
        img = tf.image.resize(img, (self.image_size[0], self.image_size[1]), tf.image.ResizeMethod.BILINEAR)
        img /= 255.

        img = tfio.experimental.color.rgb_to_lab(img)

        L = img[:, :, 0]
        L /= 100.
        a = img[:, :, 1]
        a /= 127.
        b = img[:, :, 2]
        b /= 127.

        L = tf.expand_dims(L, -1)
        a = tf.expand_dims(a, -1)
        b = tf.expand_dims(b, -1)
        #
        ab_channel = tf.concat([a, b], axis=-1)



        return (L, ab_channel)



    @tf.function
    def preprocess(self, sample):
        img = tf.cast(sample['image'], tf.float32)
        img = tf.image.resize(img, (self.image_size[0], self.image_size[1]), tf.image.ResizeMethod.BILINEAR)
        img /= 255.

        img = tfio.experimental.color.rgb_to_lab(img)

        L = img[:, :, 0]
        L /= 100.
        a = img[:, :, 1]
        a /= 127.
        b = img[:, :, 2]
        b /= 127.

        L = tf.expand_dims(L, -1)
        a = tf.expand_dims(a, -1)
        b = tf.expand_dims(b, -1)
        #
        ab_channel = tf.concat([a, b], axis=-1)

        # data augmentation
        if tf.random.uniform([], minval=0, maxval=1) > 0.5:
            L = tf.image.flip_left_right(L)
            ab_channel = tf.image.flip_left_right(ab_channel)

        return (L, ab_channel)

    @tf.function
    def preprocess_valid(self, sample):
        img = tf.cast(sample['image'], tf.float32)
        img = tf.image.resize(img, (self.image_size[0], self.image_size[1]), tf.image.ResizeMethod.BILINEAR)
        img /= 255.

        img = tfio.experimental.color.rgb_to_lab(img)

        L = img[:, :, 0]
        L /= 100.
        a = img[:, :, 1]
        a /= 127.
        b = img[:, :, 2]
        b /= 127.

        L = tf.expand_dims(L, -1)
        a = tf.expand_dims(a, -1)
        b = tf.expand_dims(b, -1)
        #
        ab_channel = tf.concat([a, b], axis=-1)

        return (L, ab_channel)

    @tf.function
    def load_original_img(self, sample):
        img = tf.cast(sample['image'], tf.float32)
        img = tf.image.resize(img, (self.image_size[0], self.image_size[1]), tf.image.ResizeMethod.BILINEAR)
        img /= 255.

        img = tfio.experimental.color.rgb_to_lab(img)

        L = img[:, :, 0]
        L /= 100.
        a = img[:, :, 1]
        a /= 127.
        b = img[:, :, 2]
        b /= 127.

        L = tf.expand_dims(L, -1)
        a = tf.expand_dims(a, -1)
        b = tf.expand_dims(b, -1)
        #
        ab_channel = tf.concat([a, b], axis=-1)

        return (L, ab_channel)

    def get_trainData(self, train_data):
        train_data = train_data.shuffle(1600)
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