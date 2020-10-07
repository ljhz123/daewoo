import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
%matplotlib inline


def cal_time(time):
    full_time = ''.join(time[:10].split('-')) + ''.join(time[11:].split(':'))
    h = full_time[8:10]
    m = full_time[10:12]
    start_m = str(int(m)-2)
    end_m = str(int(m)+2)
    start_h, end_h = h, h
    if start_m < "0":
        start_m = str(int(start_m)+60)
        start_h = str(int(h)-1)
    if end_m > "59":
        end_m = str(int(end_m)-60)
        end_h = str(int(h)+1)

    _start = ''.join((full_time[:8], start_h, full_time[10:]))
    start = ''.join((_start[:10], start_m, _start[12:]))
    _end = ''.join((full_time[:8], end_h, full_time[10:]))
    end = ''.join((_end[:10], end_m, _end[12:]))

    return start, end


# set radar_path and load WaveParam_2020.csv
radar_path = '/media/lepoeme20/Data/projects/daewoo/brave/waveradar/WaveParam_2020.csv'
radar_df = pd.read_csv(radar_path, index_col=None)
radar_df = radar_df.rename(columns={"Date&Time": "Date"})

# set data_path
data_path = '/media/lepoeme20/Data/projects/daewoo/brave/crop'

# set folder (date)
folders = sorted(os.listdir(data_path))

total_img = list()
total_time = list()
total_label = list()
for folder in folders:
    print(folder)
    # extract specific time and empty rows
    df = radar_df[radar_df.Date.str.contains(folder[:10], case=False)]
    df = df[df.Date.str[11:13] > '06']
    df = df[df.Date.str[11:13] < '17']
    radar = df[df[' SNR'] != 0.]

    # get images
    all_imgs = sorted(os.listdir(os.path.join(data_path, folder)))
    _imgs = list(filter(lambda x: 7 <= int(x[8:10]) < 17, all_imgs))

    label_list = list()
    img_list = list()
    time_list = list()

    for idx in range(radar.shape[0]):
        time = radar['Date'].iloc[idx]
        label = radar[' T.Hs'].iloc[idx]
        start, end = cal_time(time)
        imgs = list(filter(lambda x: start <= x[:-6] <= end, _imgs))
        img_list.extend(imgs)
        label_list.extend([label]*len(imgs))
        time_list.extend([time]*len(imgs))

    total_img.extend(img_list)
    total_time.extend(time_list)
    total_label.extend(label_list)

total_img = [crop_path+img for img in total_img]
data_dict = {'time':total_time, 'image':total_img, 'label':total_label}
df = pd.DataFrame(data_dict)