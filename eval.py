import cv2 as cv
import numpy as np
import os 
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from model.DeepLab_ResNet import DeepLabv3_ResNet_plus
from model.UNet import UNet, NestedUNet
from utils.dataset import BacteriaDataset

# def eval(net, device, dataset, data_list):
#     test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

#     net.eval()
#     with torch.no_grad():
#         for i, (image, mask) in enumerate(test_loader):
#             image = image.to(device=device, dtype=torch.float32)
#             pred_mask = net(image)
#             for j in range(1):
#                 pred = pred_mask[j,:]
#                 pred = F.softmax(pred, dim=1)
#                 pred = pred_mask.argmax(1)
#                 pred = np.array(pred.data.cpu()[0])
#                 pred = np.where(pred==1, 255, pred)
#                 pred = np.where(pred==2, 128, pred)
#                 pred = np.array(pred, dtype=np.uint8)
#                 cv.imwrite(os.path.join('predict', data_list[i * 2 + j].rstrip()), pred)

def eval(net, device, save_dir, image_list, image_transforms):
    net.eval()
    with torch.no_grad():
        for image_path in image_list:
            image = Image.open(image_path)
            image = image_transforms(image)
            image = image.unsqueeze(0)
            image = image.to(device=device, dtype=torch.float32)
            pred_mask = net(image)

            pred = pred_mask[0,:]
            pred = F.softmax(pred, dim=1)
            pred = pred_mask.argmax(1)
            pred = np.array(pred.data.cpu()[0])
            pred = np.where(pred==1, 255, pred)
            pred = np.where(pred==2, 128, pred)
            pred = np.array(pred, dtype=np.uint8)

            image_raw = cv.imread(image_path)
            image_raw = cv.resize(image_raw, (256, 256))
            result = np.concatenate((image_raw, (np.stack((pred,)*3, axis=-1))), axis=1)
            cv.imwrite(os.path.join(save_dir, os.path.basename(image_path)), result)
            print(os.path.join(save_dir, os.path.basename(image_path)))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_dir = 'data'
    data_txt = 'data/test.txt'
    save_dir = 'predict'
    image_transforms = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor()
    ])

    dataset_test = BacteriaDataset(data_dir=data_dir, data_txt=data_txt, image_transforms=image_transforms, isRandom=False)

    # net = DeepLabv3_ResNet_plus(in_channels=3, n_classes=3)
    # net = NestedUNet(n_channels=3, n_classes=3)
    net = UNet(n_channels=3, n_classes=3)
    net.load_state_dict(torch.load('saved/20210506_UNet/model.pth', map_location=device))
    net.to(device=device)

    data_list = open(data_txt).readlines()
    for i in range(len(data_list)):
        data_list[i] = os.path.join(data_dir, 'images', data_list[i].rstrip())

    # eval(net=net, device=device, dataset=dataset_test, data_list=data_list)
    eval(net=net, device=device, save_dir=save_dir, image_list=data_list, image_transforms=image_transforms)

if __name__ == '__main__':
    main()