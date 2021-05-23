import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image, ImageOps

class BacteriaDataset(Dataset):
    def __init__(self, data_dir, data_txt, image_transforms, isRandom=True):
        self.data_dir = data_dir
        self.data_txt = data_txt
        self.image_transforms = image_transforms
        self.isRandom = isRandom

        self.images_list = []
        self.masks_list = []
        
        data_list = open(data_txt).readlines()
        for data in data_list:
            self.images_list.append(os.path.join(data_dir, 'images', data[:-1]))
            self.masks_list.append(os.path.join(data_dir, 'masks', data[:-1]))

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image = Image.open(self.images_list[idx])
        mask = Image.open(self.masks_list[idx])

        if self.isRandom:
            isFlip = random.random()
            if isFlip > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        mask = ImageOps.grayscale(mask)
        mask = mask.resize((256, 256), 0)
        mask = np.array(mask)

        image = self.image_transforms(image)
        mask = torch.from_numpy(mask)

        return image, mask
    
def main():
    data_dir ='D:/pytorch/Segmentation/Bacteria_Detection/data'
    data_txt ='D:/pytorch/Segmentation/Bacteria_Detection/data/train.txt'

    image_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = BacteriaDataset(data_dir, data_txt, image_transforms)
    print(dataset.__len__())

if __name__ == '__main__':
    main()