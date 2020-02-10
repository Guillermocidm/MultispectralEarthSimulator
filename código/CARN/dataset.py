import os
import glob
import random
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms

def random_crop(hr, lr, size, scale):
    h, w = lr.shape[:-1]
    x = random.randint(0, w-size)
    y = random.randint(0, h-size)

    hsize = size*scale
    hx, hy = x*scale, y*scale
    crop_lr = lr[y:y+size, x:x+size].copy()
    crop_hr = hr[hy:hy+hsize, hx:hx+hsize].copy()

    return crop_hr, crop_lr


def random_flip_and_rotate(im1, im2):
    if random.random() < 0.5:
        im1 = np.flipud(im1)
        im2 = np.flipud(im2)

    if random.random() < 0.5:
        im1 = np.fliplr(im1)
        im2 = np.fliplr(im2)

    angle = random.choice([0, 1, 2, 3])
    im1 = np.rot90(im1, angle)
    im2 = np.rot90(im2, angle)

    return im1.copy(), im2.copy()


class TrainDataset(data.Dataset):
    def __init__(self, path, size, scale):
        super(TrainDataset, self).__init__()
        self.size = size
        self.path = path
        self.hr = glob.glob(os.path.join("{}train/25cm/".format(path), "*.png"))
        self.lr = glob.glob(os.path.join("{}train/1m/".format(path),"*.png"))
        self.scale = scale
        self.hr.sort()
        self.lr.sort()
        print(len(self.lr))
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        size = self.size
        hr = Image.open(self.hr[index])
        lr = Image.open(self.lr[index])
        hr = hr.convert("RGB")
        lr = lr.convert("RGB")
        hr = np.array(hr)
        lr = np.array(lr)
        item = [random_crop(hr,lr,size,self.scale)]
        item = [random_flip_and_rotate(hr, lr) for hr, lr in item]
        
        return [(self.transform(hr), self.transform(lr)) for hr, lr in item]

    def __len__(self):
        return len(self.hr)
        

class TestDataset(data.Dataset):
    def __init__(self, dirname, scale):
        super(TestDataset, self).__init__()
        self.name  = dirname.split("/")[-1]
        self.scale = scale
        self.hr = glob.glob(os.path.join("{}test/GroundTruth/".format(dirname), "*.png"))
        self.lr = glob.glob(os.path.join("{}test/Input/".format(dirname), "*.png"))                
        self.hr.sort()
        self.lr.sort()
        self.lr = self.lr
        self.hr = self.hr
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        hr = Image.open(self.hr[index])
        lr = Image.open(self.lr[index])

        hr = hr.convert("RGB")
        lr = lr.convert("RGB")
        filename = self.hr[index].split("/")[-1]
        
        return self.transform(hr), self.transform(lr), filename

    def __len__(self):
        return len(self.hr)
