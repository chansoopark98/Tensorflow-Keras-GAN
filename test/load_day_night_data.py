import glob
import numpy as np
import os
import pandas as pd
import cv2


names = ['dirty',
'daylight',
'night',
'sunrisesunset',
'dawndusk',
'sunny',
'clouds',
'fog',
'storm',
'snow',
'warm',
'cold',
'busy',
'beautiful',
'flowers',
'spring',
'summer',
'autumn',
'winter',
'glowing',
'colorful',
'dull',
'rugged',
'midday',
'dark',
'bright',
'dry',
'moist',
'windy',
'rain',
'ice',
'cluttered',
'soothing',
'stressful',
'exciting',
'sentimental',
'mysterious',
'boring',
'gloomy',
'lush']

file_path = './test/imageAlignedLD/images/'
tsv_file_path = './test/imageAlignedLD/annotations/annotations.tsv'
img_list =  glob.glob(os.path.join(file_path, '*', '*.jpg'))
img_list.sort(reverse=True)
file = pd.read_csv(tsv_file_path, sep ='\t', names=names)

get_value = file['night']

night_save_path = './test/night/'
day_save_path = './test/day/'
os.makedirs(night_save_path, exist_ok=True)
os.makedirs(day_save_path, exist_ok=True)

for idx, value in enumerate(get_value):
    file_name = file.index[idx]
    
    value = value.split(',')
    confidence = float(value[0])

    

    path_list = img_list[idx].split('/') # ./test/imageAlignedLD/images/00017660/20120504_022444.jpg 90000014/97.jpg
    image_path = path_list[0] + '/' + path_list[1] + '/' + path_list[2] + '/' + path_list[3] + '/' + '/' + file_name
    
    print('file_name', file_name)
    print('confidence', confidence)
    
    img = cv2.imread(image_path)
    # cv2.imshow('test', img)
    # cv2.waitKey(0)

    if confidence >= 0.3:
        cv2.imwrite(night_save_path + str(idx) + '_' + str(confidence) + '.png', img)
    else:
        cv2.imwrite(day_save_path + str(idx) + '_' + str(confidence) + '.png', img)
    



