import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
%matplotlib inline

data_path = '/media/lepoeme20/Data/projects/daewoo/brave/data'
folder = '2020-05-06-0'
all_imgs = sorted(os.listdir(os.path.join(data_path, folder)))
imgs = list(filter(lambda x: 7 <= int(x[8:10]) < 17, imgs))

idx = iter(np.linspace(0, len(imgs), 30, dtype=int))
start = 0
end = len(imgs)

def show_image(idx):
    img = cv2.imread(os.path.join(data_path, folder, imgs[idx]), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)
    plt.show()
    print(os.path.join(data_path, img_idx[idx]), '\n{}/{}'.format(idx, len(img_idx)))

def show_images(idx_list):
    fig = plt.figure(figsize=(14, 20))
    for i, idx in enumerate(idx_list, 1):
        img = cv2.imread(os.path.join(data_path, img_idx[idx]), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, dsize=(0, 0), fx=2, fy=2)

        ax = fig.add_subplot(5, 2, i)
        ax.imshow(img)
        ax.set_title(img_idx[idx])
        ax.axis("off")
    plt.show()

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


show_image(next(idx))
show_images(next(idx_list))
# resizing(16) #확인용
