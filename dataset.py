from torch.utils import data
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from transforms import Transforms
import glob
from torchvision.transforms import functional


class Dataset(data.Dataset):
    def __init__(self, data_path, is_train):
        self.is_train = is_train
        if is_train:
            is_train = 'train_data'
        else:
            is_train = 'test_data'

        self.image_list = glob.glob(os.path.join(data_path, is_train, 'rgb', '*.jpg'))
        self.imageL_list = glob.glob(os.path.join(data_path, is_train, 'L', '*.jpg'))
        self.label_list = glob.glob(os.path.join(data_path, is_train, 'DMap_sigma4', '*.h5'))
        self.image_list.sort()
        self.label_list.sort()

    def __getitem__(self, index):
        image_rgb = Image.open(self.image_list[index]).convert('RGB')
        image_L = Image.open(self.imageL_list[index])
        image = [image_rgb, image_L]
        label = h5py.File(self.label_list[index], 'r')
        density = np.array(label['density'], dtype=np.float32)
        attention = np.array(label['attention'], dtype=np.float32)
        gt = np.array(label['gt'], dtype=np.float32)
        trans = Transforms((0.8, 1.2), (400, 400), 2, (0.5, 1.5))
        if self.is_train:
            image, density, attention = trans(image, density, attention)
            return image, density, attention
        else:
            height, width = image_rgb.size[1], image_rgb.size[0]
            height = round(height / 16) * 16
            width = round(width / 16) * 16
            image[0] = image[0].resize((width, height), Image.BILINEAR)
            image[1] = image[1].resize((width, height), Image.BILINEAR)

            image[0] = functional.to_tensor(image[0])
            image[0] = functional.normalize(image[0], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

            image[1] = functional.to_tensor(image[1])
            image[1] = functional.normalize(image[1], [0.485], [0.229])
            return image, gt

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    train_dataset = Dataset('./DATA', True)
    train_loader = data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    for image, label, att in train_loader:

        img = np.transpose(image.numpy().squeeze(), [1, 2, 0]) * 0.2 + 0.45
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.subplot(1, 3, 2)
        plt.imshow(label.squeeze(), cmap='jet')
        plt.subplot(1, 3, 3)
        plt.imshow(att.squeeze(), cmap='jet')
        plt.show()
