import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
%matplotlib inline

root_path = '/media/project/20/dsme/all_datasets/daewoo'

ship = ['weather', 'lngc', 'vlcc', 'lngc2', 'hyundai']
data_type = 'raw_images'
data_path = os.path.join(root_path, ship[0], data_type)
img_idx = sorted(os.listdir(data_path))

idx = iter(range(len(img_idx)))

start = 920
end = len(img_idx)
step = 10
result = [list(np.arange(start, end))[i * step:(i + 1) * step] for i in range((len(list(np.arange(start, end))) + step - 1) // step)]
idx_list = iter(result)

def show_image(idx):
    img = cv2.imread(os.path.join(data_path, img_idx[idx]), cv2.IMREAD_COLOR)
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


show_image(next(idx_list))
show_images(next(idx_list))
# resizing(16) #확인용
