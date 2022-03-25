import h5py
import scipy.io as io
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter

root = './DATA'
train_data_path = os.path.join(root,'train_data', 'rgb')
test_data_path = os.path.join(root,'test_data', 'rgb')
path_sets = [train_data_path,test_data_path]

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

for img_path in img_paths:
    print(img_path)
    mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('rgb', 'ground_truth'))
    img = plt.imread(img_path)
    k = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    gt = mat['annPoints']
    count = 0
    for i in range(len(gt)):
        if int(gt[i][1]) < img.shape[0] and int(gt[i][1]) >= 0 and int(gt[i][0]) < img.shape[1] and int(gt[i][0]) >= 0:
            k[int(gt[i][1]), int(gt[i][0])] = 1
            count += 1
    k = gaussian_filter(k, 4)
    att = k > 0.001
    att = att.astype(np.float32)
    # fig = plt.figure('dmapfig')
    # plt.imshow(k)
    # plt.savefig(fname=img_path.replace('im_gt', 'dmap'))
    with h5py.File(img_path.replace('.jpg', '.h5').replace('rgb', 'DMap_sigma4'), 'w') as hf:
        hf['density'] = k
        hf['attention'] = att
        hf['gt'] = count
