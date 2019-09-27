import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
import yaml

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms, utils

from PIL import Image

class SADataset(Dataset):

    def __init__(self, csv_file, trainorval, transform=None):
        iter_csv = pd.read_csv(csv_file, iterator=True)
        df = pd.concat([chunk[chunk['set'] == trainorval] for chunk in iter_csv])
        self.csv_file = csv_file
        self.data = df
        self.trainorval = trainorval
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_right = self.data.iloc[idx, 0]
        image_left = self.data.iloc[idx, 1]
        image_disparity = self.data.iloc[idx, 2]
        pro_path = self.data.iloc[idx, 3]


        #img_name = os.path.join(image_file1)
        #image = io.imread(image_right)
        img_rgba = Image.open(image_right)
        image_data = img_rgba.convert('RGB')

        if self.transform:
            image_data = self.transform(image_data)
        
        pro = getproprioception(self, idx, pro_path)
        if pro is not None:
            tensorpro = torch.tensor(pro)

        path = image_right
        if "baxter" in self.csv_file:
            target = 0
        else:
            target = 1
        #target = torch.tensor(target)

        #target = 'baxter' #to do: take it dynamically from folder name later

        #sample = {'image': image_data, 'target': target, 'path': path, 'pro':tensorpro }
        #return sample

        return image_data, target, path, tensorpro

#sa_dataset = SADataset(csv_file='20190925_baxter.csv', trainorval="val")#,
                                    #root_dir='/home/ali/Pytorchwork/level1/')
#fig = plt.figure()

def getfilename(self, index, path, proprioception):
    with open(path, 'r') as outputfile:
        try:
            x = outputfile.read()
        except outputfile.errors as exc:
            print(exc)
        x = yaml.load(x)
    if proprioception == "all":
        return x
    else:
        return x[proprioception]

def getvelocity(self, index, path):
    velocity = getfilename(self, index, path, "velocity")
    return velocity
               

def geteffort(self, index, path):
    effort = getfilename(self, index, path, "effort")
    return effort

def getposition(self, index, path):
    position = getfilename(self, index, path, "position")
    return position

def getproprioception(self, index, path):
    proprioception = getfilename(self, index, path, "all")
    #proprioception = proprioception['velocity'] +  proprioception['effort'] + proprioception['position'] #<class 'list'>  len=57
    proprioception = proprioception.velocity +  proprioception.effort + proprioception.position
    return proprioception

def show_data(image):
    plt.imshow(image)
    plt.pause(0.001)

""" for i in range(len(sa_dataset)):
    sample = sa_dataset[i]

    print(i, sample['image'].shape)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_data(sample['image'])

    if i == 3:
        plt.show()
        break """