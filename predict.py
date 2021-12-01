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
IMAGE_SIZE = (224, 224)
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

weight_name = '_1201_best_loss'
model.load_weights(CHECKPOINT_DIR + weight_name + '.h5',by_name=True)
model.summary()


buffer = 0
batch_index = 1
save_path = './checkpoints/results/'+SAVE_MODEL_NAME+'/'
os.makedirs(save_path, exist_ok=True)

def demo_prepare(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.keras.applications.imagenet_utils.preprocess_input(img, mode='tf')

    r = img[:, :, 0]
    r = tf.expand_dims(r, -1)

    return (r, img)

filenames = os.listdir('./demo_images')
filenames.sort()
demo_imgs = tf.data.Dataset.list_files('./demo_images/' + '*', shuffle=False)
demo_test = demo_imgs.map(demo_prepare)
demo_test = demo_test.batch(BATCH_SIZE)
demo_steps = len(filenames) // BATCH_SIZE

for r, img in tqdm(demo_test, total=demo_steps):
    pred = model.predict_on_batch(r)#pred = tf.nn.softmax(pred)


    pred = pred[0]
    pred += 1.
    pred *= 127.5
    pred = tf.cast(pred, tf.int32)

    tf.keras.preprocessing.image.save_img(save_path + str(batch_index) + '_1_input.jpg', r[0])
    tf.keras.preprocessing.image.save_img(save_path + str(batch_index) + '_2_gt.jpg', img[0])
    tf.keras.preprocessing.image.save_img(save_path + str(batch_index) + '_3_out.jpg', pred)

    batch_index +=1





