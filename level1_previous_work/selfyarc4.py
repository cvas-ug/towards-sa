# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# https://stackoverflow.com/questions/44146655/how-to-convert-pretrained-fc-layers-to-conv-layers-in-pytorch
# https://github.com/mortezamg63/Accessing-and-modifying-different-layers-of-a-pretrained-model-in-pytorch

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
import itertools
import time



plt.ion()   # interactive mode

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        #transforms.RandomResizedCrop(224), # (340800 size)
        transforms.Resize(224), # (86400 size)
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        #transforms.Resize(256),
        transforms.Resize(224),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

#data_dir = 'baxter_data'
#data_dir = 'copyfromboth_16-10-2018'
#data_dir = '20190220-21' # Unbalanced data
#data_dir = '20190301' # Balanced data
data_dir = '20190311' # Total 99 in /eval/baxter has 99 data of baxter(synced)
                      # Total 99 in /eval/env (49 pro and images unsynced(which represent (not me)) + (50 env)

image_datasets = {x: customdataset.ImageFolderWithPaths(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("Training runs on : " + str(device))

#torch.cuda.manual_seed(2)
#torch.manual_seed(2)

#torch.cuda.manual_seed(24)
#torch.manual_seed(24)

torch.cuda.manual_seed(2481)
torch.manual_seed(2481)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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


# Visualizing the model predictions
# Generic function to display predictions for a few images
def visualize_model(model, num_images=4):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels, path, tensorpro) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            tensorpro = tensorpro.to(device)

            outputs = model(inputs)
            outputs = conmodel(outputs, tensorpro)
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

# Training the model
# Now, let’s write a general function to train a model. Here, we will illustrate:
# Scheduling the learning rate
# Saving the best model

class Arc4(nn.Module):
    def __init__(self):
        super(Arc4, self).__init__()
        #self.fc2 = nn.Linear(19, 19)
        #self.fc3 = nn.Linear(21, 2)
        self.pool = nn.MaxPool2d(2)
        self.conv1img = nn.Conv2d(512, 256, kernel_size=2)
        self.conv2img = nn.Conv2d(256, 128, kernel_size=1)
        
        self.conv1pro = nn.Conv2d(1, 16, kernel_size=1)
        self.conv2pro = nn.Conv2d(16, 128, kernel_size=1)

        self.conv1mix = nn.Conv2d(1, 300, kernel_size=1)
        self.conv2mix = nn.Conv2d(300, 150, kernel_size=1)

        self.fc1 = nn.Linear(86400, 60) #(62400, 60) 
        self.fc2 = nn.Linear(60, 2)


    def forward(self, inputFCFromResnet, pro):
        #print("The size of image after resnet18 is : {}".format(inputFCFromResnet.shape))
        #print("The size of proprioseption is : {}".format(pro.shape))
        x = self.conv1img(inputFCFromResnet)
        x = F.relu(x)
        #x = self.pool(x)
        #print("The size of image after conv1 is : {}".format(x.shape))
        x = self.conv2img(x)
        x = F.relu(x)
        #x = self.pool(x)
        #print("The size of image after conv2 is : {}".format(x.shape))

        #pro = pro.unsqueeze(-1)
        #pro = pro.unsqueeze(0)
        pro = pro.view(1,1,1,-1)

        #print("The size of proprioseption after unsequeeze two times : {}".format(pro.shape))
        y = self.conv1pro(pro)
        y = F.relu(y)
        #y = self.pool(y)
        #print("The size of proprioseption after conv1 is : {}".format(y.shape))
        y = self.conv2pro(y)
        y = F.relu(y)
        #y = self.pool(y)
        #print("The size of proprioseption after conv2 is : {}".format(y.shape))
        x = x.view(x.size(0), -1)
        y = y.view(y.size(0), -1)
        #x = inputFCFromResnet
        concatimageandpro = torch.cat((y, x), dim=-1)
        concatimageandpro = concatimageandpro.reshape(128,-1)
        concatimageandpro = concatimageandpro.view(1,1,128,-1)
        mix = self.pool(F.relu(self.conv1mix(concatimageandpro)))
        mix = self.pool(F.relu(self.conv2mix(mix)))

        mix = mix.view(-1, 86400) # 62400)
        mix = F.relu(self.fc1(mix))
        out = self.fc2(mix)
        return out


"""         # x = self.conv4(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x, 2)

        # use this to calculate the view size
        #print(x.size())
        #exit()
        # x = x.view(-1, 18 * 28 * 28)
        #x = x.view(-1, 3456)
        x = inputFCFromResnet.view(-1, 2)
        # check if needed
        #x = F.relu(self.fc1(x))
    
        p = self.fc2(pro)
        
        # now we can reshape `c` and `f` to 2D and concat them
        #combined = torch.cat((x.view(x.size(0), -1), p.view(p.size(0), -1)), dim=1)
        combined = torch.cat((x, p.view(p.size(0), -1)), dim=1)

        out = self.fc3(combined)
        # print(x.size())

        # return F.sigmoid(x)
        return out
        #return F.log_softmax(x)"""


def train_model(model, conmodel, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    #conmodel = Arc4()
    #conmodel = conmodel.to(device)

    closs_average = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        losses = []
        closs = 0
        batch_id = 0

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
                batch_id += 1
                inputs = inputs.to(device)
                labels = labels.to(device)
                tensorpro = tensorpro.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    #print(outputs.shape)
                    #print("+"*20)
                    #print(tensorpro.shape)
                
                    outputs = conmodel(outputs, tensorpro)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    closs += loss.item()

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        if batch_id % 100 == 0:
                            losses.append(loss.item())
                            print('this Loss: {:.6f}'.format(loss.item()))
                                      #44866
                        if (batch_id % dataset_sizes['train'] == 0) and (batch_id != 0):
                            print('[%d   %d] loss: %.6f' % (epoch+1, batch_id+1, closs/dataset_sizes['train'])) #44866)) #1472))
                            closs_average.append(closs / dataset_sizes['train']) #44866) #1472)
                            closs = 0
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    t1 = np.arange(0, num_epochs, 1)
    plt.figure()
    plt.plot(t1, closs_average)# ,label='epoch'+str(epoch))
    #plt.legend(loc='upper left', mode="expanded", shadow=True, ncol=2)
    plt.xlabel("Epoch number", fontsize=12, color='blue')
    plt.ylabel("Average loss", fontsize=12, color='blue')
    plt.title("Arch4 training losses")
    print("---")
    print(closs_average)
    plt.show()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def accuracy(model):
    model.eval()
    correctHits = 0
    total = 0
    accuracy = 0
    trueSelf_predictedSelf = 0
    trueSelf_predictedEnvironment = 0
    trueEnvironment_predictedSelf = 0
    trueEnvironment_predictedEnvironment = 0
    for batches in dataloaders['val']:
        inputs, labels, path, proprioception = batches
        inputs = inputs.to(device)
        labels = labels.to(device)
        proprioception = proprioception.to(device)

        outputs = model(inputs)
        outputs = conmodel(outputs, proprioception)
        _, outputs = torch.max(outputs.data, 1)  # return max as well as its index
        total += labels.size(0)
        correctHits += (outputs == labels).sum().item()
        accuracy = (correctHits/total)*100
        val = [outputs.item(), labels.item()]
        print(val)
        if val == [0, 0]:
            trueSelf_predictedSelf += 1
        if val == [1, 0]:
            trueSelf_predictedEnvironment += 1
        if val == [0, 1]:
            trueEnvironment_predictedSelf += 1
        if val == [1, 1]:
            trueEnvironment_predictedEnvironment += 1
    print('Accuracy ={} on total of batch {}'.format(accuracy, total))
    cm = np.array([[trueSelf_predictedSelf, trueSelf_predictedEnvironment],
                [trueEnvironment_predictedSelf, trueEnvironment_predictedEnvironment]])
    
        # From self prospective:
    # self : is positive class.
    # env  : is negative class.
    # P condition positive: the number of real positive cases in the data (tp+fn)
    # N condition negative: the number of real negative cases in the data
    tp = trueSelf_predictedSelf #truly predicted positive 
    fn = trueSelf_predictedEnvironment #falsely predicted negative
    fp = trueEnvironment_predictedSelf #falsely predicted positive
    tn = trueEnvironment_predictedEnvironment #truly predicted negative
    p= tp+fn #actual self
    n= fp+tn #actual env
    print(trueSelf_predictedSelf + trueSelf_predictedEnvironment)
    print(trueEnvironment_predictedEnvironment + trueEnvironment_predictedSelf)
    print(tp)
    print(fn)
    print(fp)
    print(tn)
    #Accuracy (ACC) = Σ True positive + Σ True negative / Σ Total population
    acc = (tp + tn)/(tp+fp+fn+tn)
    print("acc = {}".format(acc))

    #Misclassification rate - overall wrong
    mis = (fn + fp)/(tp+fp+fn+tn)
    print("Misclassification rate = {}".format(mis))

    #Prevalence = Σ Condition positive / Σ Total population
    prevalence = p / (tp+fp+fn+tn)
    print("Prevalence = {}".format(prevalence))

    #sensitivity, recall, hit rate, or true positive rate (TPR)
    #when it's actually self, how often does it predict self.
    tpr = tp / p
    print("sensitivity - true positive rate (TPR) = {}".format(tpr))

    #When it is actually env, how often does it predicted self.
    #predicted as self, but actually it is env
    fpr = fp / n
    print("false alarm - False positive rate (FPR) = {}".format(fpr))

    #When it is actually env, how often does it predicted env (TNR).
    #Specificity
    tnr = tn / n
    print("Specificity - true nigative rate (TNR) = {}".format(tnr))

    #when predict self, how often it is correct.
    #Positive predictive value (PPV), precision
    ppv = tp / (tp+fp)
    print("precision - Positive predictive value (PPV) = {}".format(ppv))

    confusionMatrix(cm, accuracy, total)


def confusionMatrix(cm, accuracy, total):
    target_names = ['Self', 'Environment']
    title = "Confusion Matrix Arch4" + ':Accuracy ={} on total of batch {}'.format(accuracy, total)
    cmap = "Greens"

    if cmap is None:
        cmap = plt.get_cmap("Blues")

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=9, color='red')
    plt.colorbar()
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)  #, rotation=45)
        plt.yticks(tick_marks, target_names)
    
    thresh = cm.max() / 1.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:,}".format(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label', fontsize=12, color='blue')
    plt.xlabel('Predicted label', fontsize=12, color='blue')
    plt.show()


if __name__ == '__main__':
    start = time.time()
    # Get a batch of training data and show it
    inputs, classes, path, tensorpro = next(iter(dataloaders['train']))
    print("here loaded")
    
    print(path)
    # Make a grid from batch    
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])

    
    # Finetuning the convnet
    # Load a pretrained model and reset final fully connected layer.
    model_ft = models.resnet18(pretrained=True)
    #for param in model_ft.parameters():
    #    param.requires_grad = False
    #num_ftrs = model_ft.fc.in_features
    #model_ft.fc = nn.Linear(num_ftrs, 2)

    #list(model_ft.modules()) # to inspect the modules of your model
    model_ft = nn.Sequential(*list(model_ft.children())[:-2])
    
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    # Train and evaluate
    # It should take around 15-25 min on CPU. On GPU though, it takes less than a minute.
    
    # I define the concatenated model.
    conmodel = Arc4()
    conmodel = conmodel.to(device)
    
    
    model_arc4 = train_model(model_ft, conmodel, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)
    
    end = time.time()
    # Evaluate
    visualize_model(model_arc4) 
    # have the confusion matrix.
    accuracy(model_arc4)

    """
    # CONVNET AS FIXED FEATURE EXTRACTOR
    # Here, we need to freeze all the network except the final layer. We 
    # need to set requires_grad == False to freeze the parameters so that 
    # the gradients are not computed in backward().
    model_conv = torchvision.models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False
    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)
    model_conv = model_conv.to(device)
    criterion = nn.CrossEntropyLoss()
    # Observe that only parameters of final layer are being optimized as
    # opoosed to before.
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
    model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=25)
    visualize_model(model_conv)
    """

    plt.ioff()
    plt.show()
    print("The training only took about : " + str((end - start) / 60))