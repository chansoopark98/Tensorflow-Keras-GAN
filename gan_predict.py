import tensorflow as tf
import os
from model.ResUnet import ResUNet
from model.Unet import Unet
from gan_train import Pix2Pix
from tensorflow.keras.initializers import RandomNormal
from tqdm import tqdm
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import tensorflow_io as tfio
# LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal 

    
def predict():
    
    model = Pix2Pix()
    
    generator = model.build_generator()
    
    generator.load_weights('./checkpoints/0523/GEN512_29.h5')
    
    BATCH_SIZE = 8
    INPUT_SHAPE_GEN = (model.image_size[0], model.image_size[1], 1)
    
    patch = int(INPUT_SHAPE_GEN[0] / 2**4)
    disc_patch = (patch, patch, 1)
    
    DATASET_DIR ='./datasets'
    test_data = tfds.load('CustomCelebahq',
                        data_dir=DATASET_DIR, split='train', shuffle_files=True)

    number_test = test_data.reduce(0, lambda x, _: x + 1).numpy()
    print("Test 데이터 개수", number_test)
    steps_per_epoch = number_test // BATCH_SIZE
    test_data = test_data.map(model.predict_data_prepare)
    test_data = test_data.padded_batch(BATCH_SIZE)
    test_data = test_data.prefetch(tf.data.experimental.AUTOTUNE)

    # prepare validation dataset
    # filenames = os.listdir('./demo_images')
    # filenames.sort()
    # demo_imgs = tf.data.Dataset.list_files('./demo_images/' + '*', shuffle=False)
    # demo_test = demo_imgs.map(model.predict_data_prepare)
    # demo_test = demo_test.batch(1)
    # demo_steps = len(filenames) // 1
    
    save_results_path = './predict_outputs/'
    
    os.makedirs(save_results_path, exist_ok=True)
    index = 1
    
    pbar = tqdm(test_data, total=steps_per_epoch, desc='Batch', leave=True, disable=False)

    for l_channel, ab_channel, norm_rgb in pbar:
        pred_ab = generator.predict(l_channel)
        
        pred_lab = tf.concat([l_channel, pred_ab], axis=-1)
        
        for i in range(len(pred_lab)):
                
                rgb = model.lab_to_rgb(lab=pred_lab[i])
                
                plt.imshow(rgb)

                plt.savefig(save_results_path + '/' + str(index)+'.png', dpi=200)
                index +=1
    
                    
                    
if __name__ == '__main__':
    
    predict()