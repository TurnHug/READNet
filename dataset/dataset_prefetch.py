import os
import time

from PIL import Image
import cv2
import torch
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import functional as F
import numbers
from PIL import Image
import random
import numpy as np
from PIL import ImageEnhance
import numpy as np
import random


class ImageDataTrain(data.Dataset):
    def __init__(self, data_root, trainsize, means, stds):
        time_s = time.time()
        self.sal_root = data_root
        self.trainsize = trainsize
        image_root = data_root + 'Image/'
        gt_root = data_root + 'GT/'
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.sal_num = len(self.images)
        self.images_buffer = []
        self.gts_buffer = []
        for i in range(self.sal_num):
            image = self.load_image(self.images[i])
            gt = self.load_sal_label(self.gts[i])
            image, gt = cv_random_flip(image, gt)
            image, gt = randomCrop(image, gt)
            image, gt = randomRotation(image, gt)
            image = colorEnhance(image)
            image = image.resize((self.trainsize, self.trainsize))
            image = ((np.array(image)/255 - means) / stds).astype(np.float32)
            gt = gt.resize((self.trainsize, self.trainsize))
            self.images_buffer.append(image)
            self.gts_buffer.append(gt)
        time_e = time.time()
        print('time cost for prefetch data: ', time_e - time_s)
        self.img_transform = transforms.Compose([
            # transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            # transforms.Normalize(means, stds)
        ])
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        self.gt_transform = transforms.Compose([
            # transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        time_s = time.time()
        # sal data loading
        # sal_image = self.load_image(self.images[index])
        # sal_label = self.load_sal_label(self.gts[index])
        sal_image = self.images_buffer[index]
        sal_label = self.gts_buffer[index]
        sal_image = self.img_transform(sal_image)
        sal_label = self.gt_transform(sal_label)
        sample = {'sal_image': sal_image, 'sal_label': sal_label}
        time_e = time.time()
        print('time cost for loading data: ', time_e - time_s)
        return sample

    def __len__(self):
        return self.sal_num

    def load_sal_label(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def load_image(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')


class ImageDataTest(data.Dataset):
    def __init__(self, data_root, data_list, testszie):
        self.data_root = data_root
        self.data_list = data_list
        with open(self.data_list, 'r') as f:
            self.image_list = [x.strip() for x in f.readlines()]

        self.image_num = len(self.image_list)

    def __getitem__(self, item):
        image, im_size = load_image_test(os.path.join(self.data_root, self.image_list[item]))
        image = torch.Tensor(image)

        return {'image': image, 'name': self.image_list[item % self.image_num], 'size': im_size}

    def __len__(self):
        return self.image_num


def get_loader(config, mode='train', pin=False):
    print(config['image_root'], config['train_size'])
    shuffle = False
    if mode == 'train':
        shuffle = True
        dataset = ImageDataTrain(config['image_root'], config['train_size'], config['means'], config['stds'])
        data_loader = data.DataLoader(dataset=dataset, batch_size=config['batch_size'], shuffle=shuffle,
                                      num_workers=config['num_workers'], pin_memory=pin)
    else:
        dataset = ImageDataTest(config['test_root'], config['test_list'], config['test_size'])
        data_loader = data.DataLoader(dataset=dataset, batch_size=config['batch_size'], shuffle=shuffle,
                                      num_workers=config['num_thread'], pin_memory=pin)
    return data_loader


# def rgb_loader(self, path):

#

def load_image_test(path):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = cv2.imread(path)
    in_ = np.array(im, dtype=np.float32)
    im_size = tuple(in_.shape[:2])
    in_ -= np.array((104.00699, 116.66877, 122.67892))
    in_ = in_.transpose((2, 0, 1))
    return in_, im_size


# several data augmentation strategies
def cv_random_flip(img, label):
    flip_flag = random.randint(0, 1)
    # flip_flag2= random.randint(0,1)
    # left right flip
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    # top bottom flip
    # if flip_flag2==1:
    #     img = img.transpose(Image.FLIP_TOP_BOTTOM)
    #     label = label.transpose(Image.FLIP_TOP_BOTTOM)
    #     depth = depth.transpose(Image.FLIP_TOP_BOTTOM)
    return img, label


def randomCrop(image, label):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region)


def randomRotation(image, label):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
    return image, label


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)


if __name__ == '__main__':
    import config

    get_loader(config.config)
