import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

import cv2
from onehot import onehot

# test image reading
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# 104.00698793, 116.66876762, 122.67891434

def read_images(root, test=True):
    file_dir = root + ('/test/' if test else '/val/')
    txt_fname = file_dir + ('test.txt' if test else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    data = [os.path.join(file_dir + 'images', i + '.png') for i in images]
    label = [os.path.join(file_dir + 'labels', i + '.png') for i in images]
    return data, label

def read_image_names(root, test=True):
    file_dir = root + ('/test/' if test else '/val/')
    txt_fname = file_dir + ('test.txt' if test else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    return images

class BagDataset(Dataset):

    def __init__(self, transform=None):
        self.data_list, self.label_list = read_images('E:/03_python_file/comp5421_TASK2')
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_name = self.data_list[idx]
        imgA = cv2.imread(img_name)
        imgA = cv2.resize(imgA, (160, 160))
        label_name = self.label_list[idx]
        imgB = cv2.imread(label_name, 0)
        imgB = cv2.resize(imgB, (160, 160))
        #imgB[imgB == 6] = 100
        #imgB[imgB != 100] = 0
        #imgB[imgB == 100] = 1
        #imgB = imgB/255
        #imgB = imgB.astype('uint8')
        #imgB = onehot(imgB, 2)
        #imgB = imgB.transpose(2, 0, 1)
        imgB = torch.FloatTensor(imgB)
        # print(imgB.shape)
        if self.transform:
            imgA = self.transform(imgA)

        return imgA, imgB

name_list = read_image_names('E:/03_python_file/comp5421_TASK2')
bag = BagDataset(transform)

# train_size = len(bag)
# test_size = len(bag)
# train_dataset, test_dataset = random_split(bag, [train_size, test_size])
# print(train_size)
# print(test_size)
#train_dataloader = DataLoader(test_dataset, batch_size=50)
#test_dataloader = DataLoader(bag, batch_size=4, shuffle=True, num_workers=4)
test_dataloader = DataLoader(bag, batch_size=1, shuffle=False, num_workers=1)

#test_dataloader = DataLoader(test_dataset, batch_size=50)

if __name__ == '__main__':

    for index, (img, label) in enumerate(test_dataloader):
        print(img.shape, label.shape)
        print(type(img))
        plt.subplot(1, 2, 1)
        output_np = img.cpu().detach().numpy().copy()  # output_np.shape = (4, 2, 160, 160)
        plt.imshow(np.squeeze(output_np[0, 0, ...]))
        # im2display = img[0, :, :, :].transpose((1, 2, 0))
        # plt.imshow(im2display, interpolation='nearest')
        # plt.imshow(img[0, 0, :, :])
        print(label.shape)
        plt.subplot(1, 2, 2)
        plt.imshow(label[0, :, :])
        plt.show()
        print(label[0, :, :])
        break

    # for train_batch in train_dataloader:
    #     print(train_batch)
    #     break

    # for test_batch in test_dataloader:
    #     print(test_batch)
    #     break
