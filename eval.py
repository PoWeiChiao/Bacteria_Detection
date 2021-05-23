import cv2 as cv
import numpy as np
import os 
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from model.DeepLab_ResNet import DeepLabv3_ResNet_plus
from model.UNet import UNet, NestedUNet
from utils.dataset import BacteriaDataset

def eval(net, device, dataset, data_list):
    test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    net.eval()
    with torch.no_grad():
        for i, (image, mask) in enumerate(test_loader):
            image = image.to(device=device, dtype=torch.float32)
            pred_mask = net(image)
            for j in range(1):
                pred = pred_mask[j,:]
                pred = F.softmax(pred, dim=1)
                pred = pred_mask.argmax(1)
                pred = np.array(pred.data.cpu()[0])
                pred = np.where(pred==1, 255, pred)
                pred = np.where(pred==2, 128, pred)
                pred = np.array(pred, dtype=np.uint8)
                cv.imwrite(os.path.join('predict', data_list[i * 2 + j][:-1]), pred)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_dir = 'data'
    data_txt = 'data/test.txt'
    image_transforms = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor()
    ])

    dataset_test = BacteriaDataset(data_dir=data_dir, data_txt=data_txt, image_transforms=image_transforms, isRandom=False)

    # net = DeepLabv3_ResNet_plus(in_channels=3, n_classes=3)
    net = NestedUNet(n_channels=3, n_classes=3)
    # net = UNet(n_channels=3, n_classes=3)
    net.load_state_dict(torch.load('model.pth', map_location=device))
    net.to(device=device)

    data_list = open(data_txt).readlines()

    eval(net=net, device=device, dataset=dataset_test, data_list=data_list)

if __name__ == '__main__':
    main()