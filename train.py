import glob
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model.DeepLab_ResNet import DeepLabv3_ResNet_plus
from model.UNet import UNet, NestedUNet
from utils.dataset import BacteriaDataset
from utils.DiceLoss import MultiClassDiceLoss
from utils.logger import Logger

def train(net, device, dataset, batch_size=1, epochs=50, lr=0.00001):
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.RMSprop(params=net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    best_loss = float('inf')
    log_train = Logger('log_train.txt')

    for epoch in range(epochs):
        train_loss = 0.0
        print('running epoch: {}'.format(epoch))
        net.train()
        for image, mask in tqdm(train_loader):
            image = image.to(device=device, dtype=torch.float32)
            mask = mask.to(device=device, dtype=torch.long)

            pred = net(image)
            loss = criterion(pred, mask)
            train_loss += loss.item() * image.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss = train_loss / len(train_loader.dataset)
        print('\tTraining Loss: {:.6f}'.format(train_loss))
        log_train.write_line(str(epoch) + ' ' + str(round(train_loss, 6)))
        if train_loss <= best_loss:
            best_loss = train_loss
            torch.save(net.state_dict(), 'model.pth')
            print('model saved')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: {}'.format(device))

    data_dir = 'data'
    data_txt = 'data/train.txt'
    image_transforms = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor()
    ])

    dataset_train = BacteriaDataset(data_dir=data_dir, data_txt=data_txt, image_transforms=image_transforms)

    # net = DeepLabv3_ResNet_plus(in_channels=3, n_classes=3)
    net = NestedUNet(n_channels=3, n_classes=3)
    # net = UNet(n_channels=3, n_classes=3)
    if os.path.isfile('model.pth'):
        net.load_state_dict(torch.load('model.pth', map_location=device))
    net.to(device=device)

    train(net=net, device=device, dataset=dataset_train, batch_size=4)

if __name__ == '__main__':
    main()