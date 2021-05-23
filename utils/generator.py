import glob
import os
import random
from logger import Logger

def main():
    image_dir = 'D:/pytorch/Segmentation/Bacteria_Detection/data/images'
    images_list = glob.glob(os.path.join(image_dir, '*.png'))
    random.shuffle(images_list)

    log_train = Logger('D:/pytorch/Segmentation/Bacteria_Detection/data/train.txt')
    log_test = Logger('D:/pytorch/Segmentation/Bacteria_Detection/data/test.txt')

    for i in range(len(images_list)):
        if i < int(len(images_list) * 0.7):
            log_train.write_line(os.path.basename(images_list[i]))
        else:
            log_test.write_line(os.path.basename(images_list[i]))

if __name__ == '__main__':
    main()
