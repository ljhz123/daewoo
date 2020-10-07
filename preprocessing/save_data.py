import os
import cv2
import numpy as np
import pandas as pd
import multiprocessing
from multiprocessing import Pool

def preprocessing(img_name):
    img = cv2.imread(os.path.join(data_path, folder, img_name), cv2.IMREAD_GRAYSCALE)
    img_gray = img.astype(np.float32)

    # ROI cropping
    img = img_gray[img_gray.shape[0]-448:, img_gray.shape[1]-448:]
    # Resizing
    img = cv2.resize(img, dsize=(224, 224))

    cv2.imwrite(os.path.join(save_path, folder, img_name), img)


if __name__=='__main__':
    df = pd.read_csv('./brave.csv').iloc[:, 1:5]
    remove_img_dict = df.to_dict()

    # set data_path
    data_path = '/media/lepoeme20/Data/projects/daewoo/brave/data'
    save_path = '/media/lepoeme20/Data/projects/daewoo/brave/crop'

    for idx in range(len(df)):
        folder = remove_img_dict['folder_name'][idx]
        start = remove_img_dict['remove_start_image'][idx]
        start = start if len(start) == 17 else start.replace('.jpg', '')
        end = remove_img_dict['remove_end_image'][idx]
        end = end if len(end) == 17 else end.replace('.jpg', '')

        # create save path
        os.makedirs(os.path.join(save_path, folder), exist_ok=True)

        # get images
        all_imgs = sorted(os.listdir(os.path.join(data_path, folder)))
        imgs = list(filter(lambda x: 7 <= int(x[8:10]) < 17, all_imgs))

        # remove images
        if start != '-' and end != '-':
            imgs = list(filter(lambda x: start <= x[:-4] < end, imgs))
        with Pool(16) as p:
            print(p.map(preprocessing, imgs))