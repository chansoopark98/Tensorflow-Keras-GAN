from utils.datasets import Dataset
from model.pix2pix import Pix2Pix
import tensorflow as tf
import argparse
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser()

parser.add_argument("--model_prefix",     type=str,   help="Model name", default='320_180_test1_sigle_GPU_bs8')
parser.add_argument("--batch_size",     type=int,   help="배치 사이즈값 설정", default=2)
parser.add_argument("--epoch",          type=int,   help="에폭 설정", default=50)
parser.add_argument("--lr",             type=float, help="Learning rate 설정", default=0.001)
parser.add_argument("--optimizer",     type=str,   help="Optimizer", default='adam')
parser.add_argument("--model_name",     type=str,   help="저장될 모델 이름",
                    default=str(time.strftime('%m%d', time.localtime(time.time()))))
parser.add_argument("--dataset_dir",    type=str,   help="데이터셋 다운로드 디렉토리 설정", default='./datasets/')
parser.add_argument("--result_dir",    type=str,   help="Validation 결과 이미지 저장 경로", default='./result_dir/')
parser.add_argument("--checkpoint_dir", type=str,   help="모델 저장 디렉토리 설정", default='./checkpoints/')
parser.add_argument("--tensorboard_dir",  type=str,   help="텐서보드 저장 경로", default='tensorboard/')
parser.add_argument("--use_weightDecay",  type=bool,  help="weightDecay 사용 유무", default=False)
parser.add_argument("--save_weight",  type=bool, help="학습 가중치 저장", default=True)
parser.add_argument("--save_frac",  type=int, help="학습 가중치 저장", default=3)
parser.add_argument("--load_weight",  help="가중치 로드", action='store_true')
parser.add_argument("--mixed_precision",  type=bool,  help="mixed_precision 사용", default=False)

args = parser.parse_args()

if __name__ == '__main__':
    CURRENT_DATE = str(time.strftime('%m%d', time.localtime(time.time()))) + '_' + args.model_prefix
    WEIGHTS_GEN = args.checkpoint_dir + '/' + CURRENT_DATE + '/GEN'
    WEIGHTS_DIS = args.checkpoint_dir + '/' + CURRENT_DATE + '/DIS'
    WEIGHTS_GAN = args.checkpoint_dir + '/' + CURRENT_DATE + '/GAN'

    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(WEIGHTS_GEN, exist_ok=True)
    os.makedirs(WEIGHTS_DIS, exist_ok=True)
    os.makedirs(WEIGHTS_GAN, exist_ok=True)

    config = {
        'image_size': (512, 512),
        'gen_input_channel': 1,
        'gen_output_channel': 2,
        'dis_input_channel': 3
    }

    gan = Pix2Pix(
        args = args,
        image_size = config['image_size'],
        gen_input_channel=config['gen_input_channel'],
        gen_output_channel=config['gen_output_channel'],
        dis_input_channel=config['dis_input_channel'],
    )

    dataset_config = Dataset(
        data_dir=args.dataset_dir,
        image_size=config['image_size'],
        batch_size=args.batch_size,
    )

    train_data = dataset_config.get_trainData(dataset_config.train_data)
    steps_per_epoch = dataset_config.number_train // args.batch_size

    valid_data = dataset_config.get_validData(dataset_config.valid_data)
    valid_per_epoch = dataset_config.number_valid // args.batch_size


    for epoch in range(args.epoch):
        pbar = tqdm(train_data, total=steps_per_epoch, desc='Batch', leave=True, disable=False)
        valid_pbar = tqdm(valid_data, total=valid_per_epoch, desc='Valid Batch', leave=True, disable=False)

        #  Train Session
        for l_channel, ab_channel, _ in pbar:
                # ---------------------
                #  Train Discriminator
                # ---------------------

                original_lab = tf.concat([l_channel, ab_channel], axis=-1)
                
                pred_ab = gan.gen_model.predict(l_channel)
                
                pred_lab = tf.concat([l_channel, pred_ab], axis=-1)

                # Unfreeze the discriminator
                gan.d_model.trainable = True
                
                fake_y_dis, real_y_dis = dataset_config.generate_patch_labels(
                    batch_size=args.batch_size,
                    disc_patch=gan.disc_patch,
                    random_augment=True)
                
                d_real = gan.d_model.train_on_batch(original_lab, real_y_dis)
                d_fake = gan.d_model.train_on_batch(pred_lab, fake_y_dis)
                
                dis_res = 0.5 * tf.add(d_fake, d_real)

                # Freeze the discriminator
                gan.d_model.trainable = False
                
                # ---------------------
                #  Train Generator
                # ---------------------
                gan_res = gan.gan_model.train_on_batch(l_channel, [real_y_dis, ab_channel])
                
                pbar.set_description("Epoch : %d Dis loss: %f, Dis ACC: %f, Gan loss: %f, Gen loss: %f Gan ACC: %f Gen MAE: %f" % (epoch,
                                            dis_res[0],
                                            dis_res[1],
                                            gan_res[0],
                                            gan_res[1],
                                            gan_res[2],
                                            gan_res[3] * 100))


        # Save Training Weights
        if args.save_weight:
            if epoch % args.save_frac == 0:
                gan.gen_model.save_weights(WEIGHTS_GEN + '_'+ str(epoch) + '.h5', overwrite=True)
                gan.d_model.save_weights(WEIGHTS_DIS + '_'+ str(epoch) + '.h5', overwrite=True)
                gan.gan_model.save_weights(WEIGHTS_GAN + '_'+ str(epoch) + '.h5', overwrite=True)
        

        #  Valid Session
        for l_channel, ab_channel, original_img in valid_pbar:
            # ---------------------
            #  Predict ab channels
            # ---------------------
            pred_ab = gan.gen_model.predict(l_channel)

            pred_lab = tf.concat([l_channel, pred_ab], axis=-1)

            rgb = dataset_config.lab_to_rgb(pred_lab, dim=4)

            mae_percent = gan.calc_metric(
                y_true=original_img, y_pred=rgb, method='mae')

            valid_pbar.set_description("Valid mae: %f" % (
                                            mae_percent * 100))
