import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
%matplotlib inline

radar_path = '/media/lepoeme20/Data/projects/daewoo/brave/waveradar/WaveParam_2020.csv'
radar_df = pd.read_csv(radar_path, index_col=None).iloc[:, :2]
radar_df = radar_df.rename(columns={"Date&Time": "Date"})

data_path = '/media/lepoeme20/Data/projects/daewoo/brave/data'
folder = '2020-05-18-1'

df = radar_df[radar_df.Date.str.contains(folder[:10], case=False)]
df = df[df.Date.str[11:13] > '06']
df = df[df.Date.str[11:13] < '17']
empty_radar = df[df[' SNR'] == 0.]

all_imgs = sorted(os.listdir(os.path.join(data_path, folder)))
imgs = list(filter(lambda x: 7 <= int(x[8:10]) < 17, all_imgs))

idx = iter(np.linspace(0, len(imgs), 15, dtype=int))
start = 0
end = len(imgs)

def show_image(idx):
    img = cv2.imread(os.path.join(data_path, folder, imgs[idx]), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)
    plt.show()
    print(os.path.join(data_path, imgs[idx]), '\n{}/{}'.format(idx, len(imgs)))

def resizing(idx):
    img = cv2.imread(os.path.join(data_path, img_idx[idx]), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[400:600, 400:]
    img = cv2.resize(img, dsize=(224, 224))
    plt.imshow(img)
    plt.show()

def get_image(idx):
    img = cv2.imread(os.path.join(data_path, img_idx[idx]), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[400:600, 400:]
    img = cv2.resize(img, dsize=(224, 224))
    return img

def get_empty_radar_images():
    time_list = [date[11:16] for date in empty_radar['Date']]
    for time in time_list:
        h = time[0:2]
        start = str(int(time[3:])-5)
        if start=='-5':
            start = '55'
            start_h = str(int(h)-1)
            start_h = start_h if len(start_h) == 2 else '0'+start_h
            start = start if len(start) == 2 else '0'+start
            end = str(int(time[3:])+5)
            end = end if len(end) == 2 else '0'+end
            if start_h == '06':
                start_name = ''.join(folder.split('-')[:3]) + '0700'
            else:
                start_name = ''.join(folder.split('-')[:3]) + start_h + start
            end_name = ''.join(folder.split('-')[:3]) + h + end
        else:
            start = start if len(start) == 2 else '0'+start
            end = str(int(time[3:])+5)
            end = end if len(end) == 2 else '0'+end
            start_name = ''.join(folder.split('-')[:3]) + h + start
            end_name = ''.join(folder.split('-')[:3]) + h + end
        try:
            start_img = [s for s in imgs if start_name in s][0]
            end_img = [s for s in imgs if end_name in s][0]
            print("Time: {} \n start: {}\n end: {}\n".format(time, start_img, end_img))
        except IndexError:
            print(IndexError)

get_empty_radar_images()
show_image(next(idx))
show_image(18141)
# resizing(16) #확인용
