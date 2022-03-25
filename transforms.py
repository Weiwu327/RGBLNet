from PIL import Image
import numpy as np
import cv2
import random
from torchvision.transforms import functional


class Transforms(object):
    def __init__(self, scale, crop, stride, gamma):
        self.scale = scale
        self.crop = crop
        self.stride = stride
        self.gamma = gamma

    def __call__(self, image, density, attention):
        # random resize
        height, width = image[0].size[1], image[0].size[0]
        if height < width:
            short = height
        else:
            short = width
        if short < 512:
            scale = 512 / short
            height = round(height * scale)
            width = round(width * scale)
            image[0] = image[0].resize((width, height), Image.BILINEAR)
            image[1] = image[1].resize((width, height), Image.BILINEAR)
            density = cv2.resize(density, (width, height), interpolation=cv2.INTER_LINEAR) / scale / scale
            attention = cv2.resize(attention, (width, height), interpolation=cv2.INTER_LINEAR)

        scale = random.uniform(self.scale[0], self.scale[1])
        height = round(height * scale)
        width = round(width * scale)
        image[0] = image[0].resize((width, height), Image.BILINEAR)
        image[1] = image[1].resize((width, height), Image.BILINEAR)
        density = cv2.resize(density, (width, height), interpolation=cv2.INTER_LINEAR) / scale / scale
        attention = cv2.resize(attention, (width, height), interpolation=cv2.INTER_LINEAR)

        # random crop
        h, w = self.crop[0], self.crop[1]
        dh = random.randint(0, height - h)
        dw = random.randint(0, width - w)
        image[0] = image[0].crop((dw, dh, dw + w, dh + h))
        image[1] = image[1].crop((dw, dh, dw + w, dh + h))
        density = density[dh:dh + h, dw:dw + w]
        attention = attention[dh:dh + h, dw:dw + w]

        # random flip
        if random.random() < 0.5:
            image[0] = image[0].transpose(Image.FLIP_LEFT_RIGHT)
            image[1] = image[1].transpose(Image.FLIP_LEFT_RIGHT)
            density = density[:, ::-1]
            attention = attention[:, ::-1]

        # random gamma
        if random.random() < 0.3:
            gamma = random.uniform(self.gamma[0], self.gamma[1])
            image[0] = functional.adjust_gamma(image[0], gamma)
            image[1] = functional.adjust_gamma(image[1], gamma)

        image[0] = functional.to_tensor(image[0])
        image[0] = functional.normalize(image[0], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        image[1] = functional.to_tensor(image[1])
        image[1] = functional.normalize(image[1], [0.485], [0.229])

        density = cv2.resize(density, (density.shape[1] // self.stride, density.shape[0] // self.stride),
                             interpolation=cv2.INTER_LINEAR) * self.stride * self.stride
        attention = cv2.resize(attention, (attention.shape[1] // self.stride, attention.shape[0] // self.stride),
                               interpolation=cv2.INTER_LINEAR)

        density = np.reshape(density, [1, density.shape[0], density.shape[1]])
        attention = np.reshape(attention, [1, attention.shape[0], attention.shape[1]])

        return image, density, attention
