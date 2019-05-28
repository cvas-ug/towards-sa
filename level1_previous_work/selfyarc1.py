# Disscussed that to skip this Arch1 as it seems that Arch1 has no potential (Disscussed with gerardo).

import torch
from torchvision import transforms
import os
import customdataset

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


data_dir = "copyfromboth_16-10-2018"
image_datasets = {x: customdataset.ImageFolderWithPaths(
                                   os.path.join(data_dir, x),
                                   data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                              shuffle=True, num_workers=0)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes


# instantiate the dataset and dataloader

#dataset = ImageFolderWithPaths(data_dir) # our custom dataset
#dataloader = torch.utils.data.DataLoader(dataset)

# iterate over data
# for inputs, labels in dataloader:
if __name__ == '__main__':
    for inputs, labels, path, pro in dataloaders['train']:
        print(inputs, labels, path, pro)
        print(type(pro))
        print(type(inputs))
        print(pro.size())
        print(inputs.size())
        third_tensor = torch.cat(pro, inputs)
        #third_tensor = pro + inputs
        print(third_tensor.size())
        print("=================================")
        

#for sample, target in enumerate(dataloaders):
    # use the above variables freely
    #print(sample, target)
    #print("ok")