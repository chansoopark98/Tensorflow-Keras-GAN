import tensorflow_io as tfio
import tensorflow_datasets as tfds
import tensorflow as tf

AUTO = tf.data.experimental.AUTOTUNE

class Dataset:
    def __init__(self, data_dir, image_size, batch_size, dataset='CustomCelebahq'):
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

        self.train_data, self.number_train = self._load_train_datasets()
        self.valid_data, self.number_valid = self._load_valid_datasets()


    def _load_valid_datasets(self):
        valid_data = tfds.load(self.dataset_name,
                               data_dir=self.data_dir, split='train[:1%]')

        number_valid = valid_data.reduce(0, lambda x, _: x + 1).numpy()
        print("검증 데이터 개수:", number_valid)

        return valid_data, number_valid


    def _load_train_datasets(self):
        # train -> 30000
        # train[:1%] -> 300
        # train[1%:] -> 29700
        train_data = tfds.load(self.dataset_name,
                               data_dir=self.data_dir, split='train[1%:]')

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


    @tf.function
    def prepare_train_ds(self, sample):
        img = sample['image']
        # data augmentation
        
        if tf.random.uniform([], minval=0, maxval=1) > 0.5:
            img = tf.image.flip_left_right(img)
            
        
        scale = tf.random.uniform([], 0.5, 1.5)
        
        nh = self.image_size[0] * scale
        nw = self.image_size[1] * scale

        img = tf.image.resize(img, (nh, nw), method=tf.image.ResizeMethod.BILINEAR)
        img = tf.image.resize_with_crop_or_pad(img, self.image_size[0], self.image_size[1])
        
        l_channel, ab_channel = self.rgb_to_lab(rgb=img)
        
        norm_rgb = (img / 128) - 1
        
        return (l_channel, ab_channel, norm_rgb)

    
    @tf.function
    def prepare_valid_ds(self, sample):
        img = sample['image']

        img = tf.image.resize(img, (self.image_size[0], self.image_size[1]), method=tf.image.ResizeMethod.BILINEAR)
        l_channel, ab_channel = self.rgb_to_lab(rgb=img)
        norm_rgb = (img / 128) - 1

        return (l_channel, ab_channel, norm_rgb)



    def rgb_to_lab(self, rgb):
        """
        Convert to rgb image to lab image

        Args:
            rgb (Tensor): (H, W, 3)

        Returns:
            Normalized lab image
            {
                Value Range
                L : -1 ~ 1
                ab : -1 ~ 1
            }
            L, ab (Tensor): (H, W, 1), (H, W, 2)
        """
        # normalize image 0 ~ 1.
        rgb /= 255. 
        
        # Convert to float32 data type.
        rgb = tf.cast(rgb, tf.float32)
        
        # Convert to rgb to lab
        lab = tfio.experimental.color.rgb_to_lab(rgb)

        l_channel = lab[:, :, 0]
        a_channel = lab[:, :, 1]
        b_channel = lab[:, :, 2]
        
        l_channel = tf.expand_dims(l_channel, axis=-1)
        a_channel = tf.expand_dims(a_channel, axis=-1)
        b_channel = tf.expand_dims(b_channel, axis=-1)

        ab_channel = tf.concat([a_channel, b_channel], axis=-1)
        
        # -1 ~ 1 scaling
        l_channel = (l_channel - 50.) / 50. 
        ab_channel /= 128.       
        
        return l_channel, ab_channel


    def lab_to_rgb(self, lab, dim=3):
        """
        Convert to lab image to rgb image

        Args:
            lab (Tensor, float32): (H, W, 3)

        Returns:
            {
                Normalized lab image
                Value Range : 0 ~ 1
            }
            RGB (Tensor, float32) :(H, W, 3)
        """      
        if dim == 4:
            batch_l = lab[:, :, :, 0]
            batch_a = lab[:, :, :, 1]
            batch_b = lab[:, :, :, 2]
        else:
            batch_l = lab[:, :, 0]
            batch_a = lab[:, :, 1]
            batch_b = lab[:, :, 2]

        
        batch_l = (batch_l * 50) + 50.
        batch_a *= 128.
        batch_b *= 128.
        
        batch_l = tf.expand_dims(batch_l, axis=-1)
        batch_a = tf.expand_dims(batch_a, axis=-1)
        batch_b = tf.expand_dims(batch_b, axis=-1)
        
        batch_lab = tf.concat([batch_l, batch_a, batch_b], axis=-1)

        rgb = tfio.experimental.color.lab_to_rgb(batch_lab)
        
        return rgb


    def generate_patch_labels(self, batch_size: int, disc_patch, random_augment: bool):
        fake_y_dis = tf.zeros((batch_size,) + disc_patch)
        real_y_dis = tf.ones((batch_size,) + disc_patch)

        if random_augment:
            if tf.random.uniform([]) < 0.05:
                real_factor = tf.random.uniform([], minval=0.8, maxval=1.)
                real_y_dis *= real_factor

        return fake_y_dis, real_y_dis


    def get_trainData(self, train_data):
        train_data = train_data.shuffle(1024)
        train_data = train_data.map(self.prepare_train_ds, num_parallel_calls=AUTO)
        # train_data = train_data.padded_batch(self.batch_size)
        train_data = train_data.batch(self.batch_size)
        train_data = train_data.prefetch(AUTO)

        return train_data

    def get_validData(self, valid_data):
        valid_data = valid_data.map(self.prepare_valid_ds, num_parallel_calls=AUTO)
        valid_data = valid_data.batch(self.batch_size).prefetch(AUTO)
        return valid_data