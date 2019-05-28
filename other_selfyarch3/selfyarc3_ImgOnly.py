
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

import customdataset
import torch.nn.functional as F

plt.ion()   # interactive mode

data_transforms = {
    'train': transforms.Compose([
        #transforms.RandomResizedCrop(224),
        transforms.Resize([224, 224]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        #transforms.Resize(256),
        #transforms.CenterCrop(224),
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

#data_dir = '20190401' #Train 2000 Baxter / 2000 Env 
                      #val 400 Baxter / 400 Env
#data_dir = '20190405' #Train 2000 Baxter / 2000 Env
                       #val  0 Baxter / 400 Env (400 Baxter (unsynced)) (the real env files deleted just to make anaysis easier)
#data_dir = '20190415' #Train 2000 Baxter / 2000 Env
                       #val  0 Baxter / 400 Env (400 Baxter (unsynced)) (the real env files deleted just to make anaysis easier)
                       #The randomization of pro file is based on the range of the env pro files.

#data_dir = '20190514'
#data_dir = '20190514-case3'
data_dir = '20190514-case4'

image_datasets = {x: customdataset.ImageFolderWithPaths(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Training runs on : " + str(device))

#current_seed = 2
#current_seed = 24
current_seed = 2481
torch.cuda.manual_seed(current_seed)
torch.manual_seed(current_seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("Current seed: {}".format(current_seed))
print("dataset     : " + data_dir)

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(3)  # pause a bit so that plots are updated

# Get a batch of training data
inputs, classes, path, tensorpro = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    all_epochs_train_losses_average = []
    all_epochs_train_accuraies_average = []
    all_epochs_val_losses_average = []
    all_epochs_val_accuraies_average = [] 

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels, path, tensorpro in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                proprioception = tensorpro.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            if phase == 'train':
                all_epochs_train_losses_average.append(epoch_loss)
                all_epochs_train_accuraies_average.append(epoch_acc.item())
            elif phase == 'val':
                all_epochs_val_losses_average.append(epoch_loss)
                all_epochs_val_accuraies_average.append(epoch_acc.item())
                

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    plotlossaverages(num_epochs, all_epochs_train_losses_average, all_epochs_val_losses_average)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def plotlossaverages(num_epochs, all_epochs_train_losses_average, all_epochs_val_losses_average):
    t1 = np.arange(0, num_epochs, 1)
    plt.figure()
    plt.plot(t1, all_epochs_train_losses_average, 'r', all_epochs_val_losses_average, 'b')#, train_accuracy_average, 'g', val_accuracy_average, 'purple' )# ,label='epoch'+str(epoch))
    plt.legend(['Train loss ave.', 'val loss ave.'])#, 'Val loss ave.', 'Val accu ave.'])
    #plt.legend(loc='upper left', mode="expanded", shadow=True, ncol=2)
    plt.xlabel("Epoch number", fontsize=12, color='blue')
    plt.ylabel("Average loss", fontsize=12, color='blue')
    plt.title("Arch3: training loss and validation loss averages")
    plt.show(5)

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels, path, tensorpro) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            proprioception = tensorpro.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

class arch3model(nn.Module):
    def __init__(self):
        super(arch3model, self).__init__()
        self.model_ft = models.resnet18(pretrained=True)
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, 19)
        self.model_ft = self.model_ft.to(device)
        
        #self.fc1 = nn.Linear(57, 128)
        #self.fc2 = nn.Linear(128, 64)
        #self.fc3 = nn.Linear(64, 32)
        #self.fc4 = nn.Linear(32, 2)
        self.fc1 = nn.Linear(19, 2)
        
    def forward(self, image):
        outof_resnet18 = self.model_ft(image)
        #pro = proprioception
        
        #x = torch.cat((outof_resnet18, proprioception), dim=1)
        #x = F.relu(self.fc1(pro))
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        #x = self.fc4(x)
        x = F.relu(self.fc1(outof_resnet18))
        return x
        

model_arc3 = arch3model()
model_arc3 = model_arc3.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(model_arc3.parameters(), lr=0.001, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
model_arc3 = train_model(model_arc3, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)
visualize_model(model_arc3)

plt.ioff()
plt.show()