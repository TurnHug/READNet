import os
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
        self.sal_root = data_root
        self.trainsize = trainsize
        image_root = data_root + '/input/'
        gt_root = data_root + '/gt/'
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.sal_num = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize(means, stds)
        ])
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        # sal data loading
        sal_image = self.load_image(self.images[index])
        sal_label = self.load_sal_label(self.gts[index])
        sal_image, sal_label = cv_random_flip(sal_image, sal_label)
        sal_image, sal_label = randomCrop(sal_image, sal_label)
        sal_image, sal_label = randomRotation(sal_image, sal_label)
        sal_image = colorEnhance(sal_image)

        sal_image = self.img_transform(sal_image)
        sal_label = self.gt_transform(sal_label)
        sample = {'sal_image': sal_image, 'sal_label': sal_label}
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


class ImageDataTrainWithSegEdge(data.Dataset):
    def __init__(self, data_root, trainsize, means, stds):
        self.sal_root = data_root
        self.trainsize = trainsize
        image_root = data_root + 'Image/'
        gt_root = data_root + 'GT/'
        seg_root = data_root + 'segany/'
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.segs = [seg_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                     or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.segs = sorted(self.segs)
        # print(self.segs)
        self.sal_num = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize(means, stds)
        ])
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        # sal data loading
        sal_image = self.load_image(self.images[index])
        sal_label = self.load_sal_label(self.gts[index])
        sal_seg = self.load_sal_label(self.segs[index])
        sal_image, sal_label, sal_seg = cv_random_flip_with_segany(sal_image, sal_label, sal_seg)
        sal_image, sal_label, sal_seg = randomCrop(sal_image, sal_label, seg=sal_seg)
        sal_image, sal_label, sal_seg = randomRotation_with_segany(sal_image, sal_label, sal_seg)
        sal_image = colorEnhance(sal_image)

        sal_image = self.img_transform(sal_image)
        sal_label = self.gt_transform(sal_label)
        sal_seg = self.gt_transform(sal_seg)
        sample = {'sal_image': sal_image, 'sal_label': sal_label, 'seg_label': sal_seg}
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


def get_loaderWithSegEdge(config, mode='train', pin=False):
    print(config['image_root'], config['train_size'])
    shuffle = False
    if mode == 'train':
        shuffle = True
        dataset = ImageDataTrainWithSegEdge(config['image_root'], config['train_size'], config['means'], config['stds'])
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
def cv_random_flip(img, label, seg=None):
    flip_flag = random.randint(0, 1)
    # flip_flag2= random.randint(0,1)
    # left right flip
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    # if seg is not None and flip_flag == 1:
    #     seg = seg.transpose(Image.FLIP_LEFT_RIGHT)
    #     return img, label, seg
    # top bottom flip
    # if flip_flag2==1:
    #     img = img.transpose(Image.FLIP_TOP_BOTTOM)
    #     label = label.transpose(Image.FLIP_TOP_BOTTOM)
    #     depth = depth.transpose(Image.FLIP_TOP_BOTTOM)
    return img, label


def cv_random_flip_with_segany(img, label, seg):
    flip_flag = random.randint(0, 1)
    # flip_flag2= random.randint(0,1)
    # left right flip
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        seg = seg.transpose(Image.FLIP_LEFT_RIGHT)
    # if seg is not None and flip_flag == 1:
    #     seg = seg.transpose(Image.FLIP_LEFT_RIGHT)
    #     return img, label, seg
    # top bottom flip
    # if flip_flag2==1:
    #     img = img.transpose(Image.FLIP_TOP_BOTTOM)
    #     label = label.transpose(Image.FLIP_TOP_BOTTOM)
    #     depth = depth.transpose(Image.FLIP_TOP_BOTTOM)
    return img, label, seg


def cv_random_flip_with_segany_edge(img, label, seg, edge):
    flip_flag = random.randint(0, 1)
    # flip_flag2= random.randint(0,1)
    # left right flip
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        seg = seg.transpose(Image.FLIP_LEFT_RIGHT)
        edge = edge.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label, seg, edge


def randomCrop(image, label, seg=None):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    if seg is not None:
        return image.crop(random_region), label.crop(random_region), seg.crop(random_region)
    return image.crop(random_region), label.crop(random_region)


def randomCrop_seg_edge(image, label, seg, edge):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region), seg.crop(random_region), edge.crop(random_region)


def randomRotation(image, label, seg=None):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        # if seg is not None:
        #     seg = seg.rotate(random_angle, mode)
        #     return image, label, seg
    return image, label


def randomRotation_with_segany(image, label, seg):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        seg = seg.rotate(random_angle, mode)
    return image, label, seg


def randomRotation_with_segany_edge(image, label, seg, edge):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        seg = seg.rotate(random_angle, mode)
        edge = edge.rotate(random_angle, mode)
    return image, label, seg, edge


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


class ImageDataTrainAll(data.Dataset):
    def __init__(self, data_root, trainsize, means, stds, any_edge=None, edge=None):
        self.sal_root = data_root
        self.trainsize = trainsize
        image_root = data_root + 'Image/'
        gt_root = data_root + 'GT/'
        self.any_edge = any_edge
        self.edge = edge
        if any_edge is not None:
            seg_root = data_root + any_edge
            self.segs = [seg_root + f for f in os.listdir(seg_root) if f.endswith('.jpg')
                         or f.endswith('.png')]
            self.segs = sorted(self.segs)
        if edge is not None:
            edge_root = data_root + edge
            self.edges = [edge_root + f for f in os.listdir(edge_root) if f.endswith('.jpg')
                          or f.endswith('.png')]
            self.edges = sorted(self.edges)
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.sal_num = len(self.images)

        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize(means, stds)
        ])
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        # sal data loading
        sal_image = self.load_image(self.images[index])
        sal_label = self.load_sal_label(self.gts[index])
        sal_any = self.load_sal_label(self.segs[index])
        sal_edge = self.load_sal_label(self.edges[index])
        sal_image, sal_label, sal_any, sal_edge = cv_random_flip_with_segany_edge(sal_image, sal_label, sal_any,
                                                                                  sal_edge)
        sal_image, sal_label, sal_any, sal_edge = randomCrop_seg_edge(sal_image, sal_label, sal_any, sal_edge)
        sal_image, sal_label, sal_any, sal_edge = randomRotation_with_segany_edge(sal_image, sal_label, sal_any,
                                                                                  sal_edge)
        sal_image = colorEnhance(sal_image)

        sal_image = self.img_transform(sal_image)
        sal_label = self.gt_transform(sal_label)
        sal_any = self.gt_transform(sal_any)
        sal_edge = self.gt_transform(sal_edge)
        sample = {'sal_image': sal_image, 'sal_label': sal_label, 'seg_label': sal_any, 'edge_label': sal_edge}
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


def get_loaderWithSegAndEdge(config, mode='train', pin=False):
    print(config['image_root'], config['train_size'])
    shuffle = True
    dataset = ImageDataTrainAll(config['image_root'], config['train_size'], config['means'], config['stds'],
                                'segany_top5/', 'Edge/')
    data_loader = data.DataLoader(dataset=dataset, batch_size=config['batch_size'], shuffle=shuffle,
                                  num_workers=config['num_workers'], pin_memory=pin)
    return data_loader


if __name__ == '__main__':
    import config

    get_loader(config.config)
