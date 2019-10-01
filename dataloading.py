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

validation_split = 0.8
current_seed = 2481
torch.cuda.manual_seed(current_seed)
torch.manual_seed(current_seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

train_csv = 'train.csv'
eval_csv = 'eval.csv'

class SADataset(Dataset):

    def __init__(self, groupset, transform=None):
        if groupset == "train":
            csv_file = train_csv
        else:
            csv_file = eval_csv
        #iter_csv = pd.read_csv(csv_file, iterator=True)
        #df = pd.concat([chunk[chunk['set'] == trainorval] for chunk in iter_csv])
        self.csv_file = csv_file
        #self.data = df
        self.data = pd.read_csv(csv_file)
        #self.data = create_datasets(self.data, groupset)
        #self.trainorval = groupset
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #image_right = self.data.dataset.values[self.data.indices][idx][0]
        image_right = self.data.iloc[idx, 0] #self.data.dataset["right"][int(self.data.indices[idx])]
        image_left = self.data.iloc[idx, 1] #self.data.dataset["left"][int(self.data.indices[idx])]
        image_disparity = self.data.iloc[idx, 2] #self.data.dataset["disparity"][int(self.data.indices[idx])]
        pro_path = self.data.iloc[idx, 3] #self.data.dataset["proprioception"][int(self.data.indices[idx])]


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
        if "env" in image_right:
            target = 1
        else:
            target = 0
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

def create_datasets(dataset, groupset):#shuffle_dataset):
    # Creating data indices for training and validation splits:
    train_size = int(validation_split * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, validation_ds = torch.utils.data.random_split(dataset, [train_size, test_size])
    print("Train size: {}".format(len(train_ds)))
    print("Validation size: {}".format(len(validation_ds)))

    #train_loader = DataLoader(train_ds, batch_size=batch_training, shuffle=shuffle_dataset, num_workers=4)
    #validation_loader = DataLoader(validation_ds, batch_size=batch_validation, shuffle=shuffle_dataset, num_workers=4)
    if groupset == "train":
        return train_ds
    else:
        return validation_ds