from PIL import Image
import glob
import os

img_data_path = '../datasets/img_align_celeba/img_align_celeba/'

train_dir = '../datasets/img_align_celeba/train'
valid_dir = '../datasets/img_align_celeba/validation'
test_dir = '../datasets/img_align_celeba/test'

cnt = 0
# train: 162770
# valid : 182637

for img in sorted(glob.glob(img_data_path+'*.jpg')):
    name = img.split('/')[-1]
    idx = name.split('.')[0]

    if int(idx) <= 162770:
        Image.open(img).save(os.path.join(train_dir, str(idx) + '.jpg'))
    elif int(idx) >= 162770 and int(idx) >= 182637:
        Image.open(img).save(os.path.join(valid_dir, str(idx) + '.jpg'))
    else:
        Image.open(img).save(os.path.join(test_dir, str(idx) + '.jpg'))