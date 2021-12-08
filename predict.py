import matplotlib.pyplot as plt
from tensorflow.keras.mixed_precision import experimental as mixed_precision
# from ddrnet_23_slim.model.model_builder import seg_model_build
from model.model_builder import base_model
import argparse
import time
import os
import tensorflow as tf
from tqdm import tqdm
from utils.datasets import Dataset
import tensorflow_io as tfio


tf.keras.backend.clear_session()

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size",     type=int,   help="배치 사이즈값 설정", default=1)
parser.add_argument("--epoch",          type=int,   help="에폭 설정", default=100)
parser.add_argument("--lr",             type=float, help="Learning rate 설정", default=0.001)
parser.add_argument("--weight_decay",   type=float, help="Weight Decay 설정", default=0.0005)
parser.add_argument("--model_name",     type=str,   help="저장될 모델 이름",
                    default=str(time.strftime('%m%d', time.localtime(time.time()))))
parser.add_argument("--dataset_dir",    type=str,   help="데이터셋 다운로드 디렉토리 설정", default='./datasets/')
parser.add_argument("--checkpoint_dir", type=str,   help="모델 저장 디렉토리 설정", default='./checkpoints/')
parser.add_argument("--tensorboard_dir",  type=str,   help="텐서보드 저장 경로", default='tensorboard')
parser.add_argument("--backbone_model", type=str,   help="EfficientNet 모델 설정", default='B0')
parser.add_argument("--train_dataset",  type=str,   help="학습에 사용할 dataset 설정 coco or voc", default='voc')
parser.add_argument("--use_weightDecay",  type=bool,  help="weightDecay 사용 유무", default=False)
parser.add_argument("--load_weight",  type=bool,  help="가중치 로드", default=False)
parser.add_argument("--mixed_precision",  type=bool,  help="mixed_precision 사용", default=True)
parser.add_argument("--distribution_mode",  type=bool,  help="분산 학습 모드 설정 mirror or multi", default='mirror')

args = parser.parse_args()
WEIGHT_DECAY = args.weight_decay
BATCH_SIZE = args.batch_size
EPOCHS = args.epoch
base_lr = args.lr
SAVE_MODEL_NAME = args.model_name
DATASET_DIR = args.dataset_dir
CHECKPOINT_DIR = args.checkpoint_dir
TENSORBOARD_DIR = args.tensorboard_dir
MODEL_NAME = args.backbone_model
TRAIN_MODE = args.train_dataset
IMAGE_SIZE = (512, 512)
num_classes = 2
USE_WEIGHT_DECAY = args.use_weightDecay
LOAD_WEIGHT = args.load_weight
MIXED_PRECISION = args.mixed_precision
DISTRIBUTION_MODE = args.distribution_mode

if MIXED_PRECISION:
    policy = mixed_precision.Policy('mixed_float16', loss_scale=1024)
    mixed_precision.set_policy(policy)

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# if use celebA dataset evaluation
# dataset = Dataset(DATASET_DIR, IMAGE_SIZE, BATCH_SIZE, mode='validation')
# test_steps = dataset.number_valid // BATCH_SIZE
# test_set = dataset.get_testData(dataset.valid_data)

model = base_model(image_size=IMAGE_SIZE, num_classes=num_classes)

# weight_name = '_1208_best_loss'
weight_name = '_1208_final_loss'
model.load_weights(CHECKPOINT_DIR + weight_name + '.h5',by_name=True)
model.summary()


buffer = 0
batch_index = 1
save_path = './demo_outputs/'+SAVE_MODEL_NAME+'/'
os.makedirs(save_path, exist_ok=True)

def demo_prepare(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize_with_pad(img, 512, 512)
    # img = tf.image.resize(img, (512, 512))

    grayscale = tfio.experimental.color.rgb_to_grayscale(img)
    gray_concat = tf.concat([grayscale, grayscale, grayscale], axis=-1)
    gray_concat /= 255.


    gray_concat = tfio.experimental.color.rgb_to_lab(gray_concat)
    L = gray_concat[:, :, 0]
    L = (L / 50.) - 1.


    return (L)

filenames = os.listdir('./demo_images')
filenames.sort()
demo_imgs = tf.data.Dataset.list_files('./demo_images/' + '*', shuffle=False)
demo_test = demo_imgs.map(demo_prepare)
demo_test = demo_test.batch(BATCH_SIZE)
demo_steps = len(filenames) // BATCH_SIZE

for r in tqdm(demo_test, total=demo_steps):

    pred = model.predict_on_batch(r)
    L_channel = r[0]
    prediction = pred[0]
    L = L_channel
    L += 1
    L *= 50.

    a = prediction[:, :, 0]
    a = (a+1) /2.
    a *= 255.
    a -= 127.

    b = prediction[:, :, 1]
    b = (b+1) /2.
    b *= 255.
    b -= 127.

    L = tf.cast(L, tf.float32)
    a = tf.cast(a, tf.float32)
    b = tf.cast(b, tf.float32)

    L = tf.expand_dims(L, -1)
    a = tf.expand_dims(a, -1)
    b = tf.expand_dims(b, -1)

    output = tf.concat([L, a, b], axis=-1)
    output = tfio.experimental.color.lab_to_rgb(output)
    new_r = output[:, :, 0]
    # new_r *= 255.
    # new_r -= 25.
    # new_r /= 255.

    new_g = output[:, :, 1]
    new_b = output[:, :, 2]
    # new_b *= 255.
    # new_b += 10.
    # new_b /= 255.

    new_r = tf.expand_dims(new_r, -1)
    new_g = tf.expand_dims(new_g, -1)
    new_b = tf.expand_dims(new_b, -1)
    new_output = tf.concat([new_r, new_g, new_b], axis=-1)

    tf.keras.preprocessing.image.save_img(save_path + str(batch_index) + '.png', output)
    batch_index+=1
    # plt.savefig(save_path + str(batch_index) + 'output.png', dpi=300)
    # pred = tf.cast(pred, tf.int32)
    # plt.imshow
    # plt.show()


