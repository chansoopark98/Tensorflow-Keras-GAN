from PIL import Image
import glob
import os

img_data_path = './test_img'


out_dir = './cvt_test_label'
cnt = 0
for img in glob.glob(test_data_path+'/*.bmp'):
    Image.open(img).save(os.path.join(out_dir, str(cnt) + '.png'))
    cnt += 1