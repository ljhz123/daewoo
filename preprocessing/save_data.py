import os
import cv2
import time
import numpy as np
import pandas as pd
import multiprocessing
from multiprocessing import Pool
from collections import defaultdict
from functools import partial

def preprocessing(img_name, data_path, folder, save_path):
    try:
        print(img_name, data_path, folder, save_path)
        img = cv2.imread(os.path.join(data_path, folder, img_name), cv2.IMREAD_GRAYSCALE)
        img_gray = img.astype(np.float32)

        # ROI cropping
        img = img_gray[img_gray.shape[0]-448:, img_gray.shape[1]-448:]
        # Resizing
        img = cv2.resize(img, dsize=(224, 224))

        cv2.imwrite(os.path.join(save_path, folder, img_name), img)
    except AttributeError:
        pass


if __name__=='__main__':
    df = pd.read_csv('./brave.csv').iloc[:, 1:5]
    remove_img_dict = defaultdict(list)

    for k, v1, v2 in zip(df.folder_name.values, df.remove_start_image.values, df.remove_end_image.values):
        tmp = (v1, v2)
        remove_img_dict[k].append(tmp)

    # set data_path
    d_path = '/media/lepoeme20/Data/projects/daewoo/brave/data'
    s_path = '/media/lepoeme20/Data/projects/daewoo/brave/crop'
    folders = sorted(os.listdir(d_path))

    for i, folder in enumerate(folders):
        # create save path
        os.makedirs(os.path.join(s_path, folder), exist_ok=True)

        # get images
        all_imgs = sorted(os.listdir(os.path.join(d_path, folder)))
        imgs = list(filter(lambda x: 7 <= int(x[8:10]) < 17, all_imgs))

        rm_imgs = []
        for (start, end) in remove_img_dict[folder]:
            start = start if len(start) == 17 else start.replace('.jpg', '')
            end = end if len(end) == 17 else end.replace('.jpg', '')
            if start != '-' and end != '-':
                rm_imgs.extend(list(filter(lambda x: start <= x[:-4] <= end, imgs)))
        imgs = list(filter(lambda x: x not in rm_imgs, imgs))

        save_img=partial(preprocessing, data_path=d_path, folder=folder, save_path=s_path)

        with Pool(16) as p:
            p.map(save_img, imgs)

        print("{}/{} \t Folder: {} | Total Image: {:,} | RM Image: {:,} | Usable Image: {:,} | time: {:.3f}s".format(
            i, len(folders), folder, len(imgs)+len(rm_imgs), len(rm_imgs), len(imgs), time.time()-start))
