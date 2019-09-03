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
import scikitplot as skplt

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
#data_dir = '20190416' #Train 2000 Baxter / 2000 Env
                       #val  0 Baxter / 400 Env (400 env (unsynced)) (the real baxter files deleted just to make anaysis easier)
                       #The randomization of pro file is based on the range of the baxter pro files.
#data_dir = '20190429' #Train 2000 Baxter / 2000 Env
                       #val  0 Baxter / 400 Env (400 env (unsynced)) (the real baxter files deleted just to make anaysis easier)
                       #The randomization of pro file is based on the range of the baxter pro files.
                       #The 2000 Env (Has about 906 images generated newlly by moving baxter hands and the scene is environment)
#--------------------------------------------------------------------------------------------------------------------------------
#data_dir = '20190514' #case1and2
#data_dir = '20190514-case3'
#data_dir = '20190514-case4'

#data_dir = '20190521' #case1and2
#data_dir = '20190521-case3'
#data_dir = '20190521-case4'

path = os.path.dirname(__file__)
print(path)
path2 = os.path.dirname(path)

#data_dir = '20190611'       #case1and2 real baxter data
#data_dir = '20190611-case3' #case3 - real baxter data
#data_dir = '20190611-case4'  #case4 - real baxter data

data_dir = '20190612'       #case1and2 real baxter data including cases3and4 in the training data.
#data_dir = '20190612-case3' #case3 - real baxter data including cases3and4 in the training data.
#data_dir = '20190612-case4' #case4 - real baxter data including cases3and4 in the training data.

data_dir =  path2 + '/' + data_dir
image_datasets = {x: customdataset.ImageFolderWithPaths(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                             shuffle=True, num_workers=0)
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
#out = torchvision.utils.make_grid(inputs)

#imshow(out, title=[class_names[x] for x in classes])

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
                    outputs = model(inputs, proprioception)  #inputs.size() torch.Size([64, 3, 224, 224]) #proprioception.size() torch.Size([64, 51])
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

            outputs = model(inputs, proprioception)
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

def eval_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_acc = 0.0

    ##all_epochs_train_losses_average = []
    ##all_epochs_train_accuraies_average = []
    all_epochs_val_losses_average = []
    all_epochs_val_accuraies_average = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['val']:
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
                    outputs = model(inputs, proprioception)  #inputs.size() torch.Size([64, 3, 224, 224]) #proprioception.size() torch.Size([64, 51])
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            ##if phase == 'train':
                ##all_epochs_train_losses_average.append(epoch_loss)
                ##all_epochs_train_accuraies_average.append(epoch_acc.item())
            #elif phase == 'val':
            if phase == 'val':
                all_epochs_val_losses_average.append(epoch_loss)
                all_epochs_val_accuraies_average.append(epoch_acc.item())
                

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    plotlossaverages_eval(num_epochs, all_epochs_val_losses_average)

    # load best model weights
    #model.load_state_dict(best_model_wts)
    return model


def plotlossaverages_eval(num_epochs, all_epochs_val_losses_average):
    t1 = np.arange(0, num_epochs, 1)
    plt.figure()
    plt.plot(t1, all_epochs_val_losses_average, 'b')#, train_accuracy_average, 'g', val_accuracy_average, 'purple' )# ,label='epoch'+str(epoch))
    plt.legend(['Training loss ave.', 'Validation loss ave.'])#, 'Val loss ave.', 'Val accu ave.'])
    #plt.legend(loc='upper left', mode="expanded", shadow=True, ncol=2)
    plt.xlabel("Epoch number", fontsize=12, color='blue')
    plt.ylabel("Average loss", fontsize=12, color='blue')
    plt.title("Arch3: training loss and validation loss averages")
    plt.show(5)

def accuracy(model):
    test_y = list()
    probas_y = list()
    prefix='./plots'

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

        #outputs = model(inputs, proprioception)
        #_, outputs = torch.max(outputs.data, 1)  # return max as well as its index
        #_, preds = torch.max(outputs, 1) #check if same above, later

        outputs_probas = model(inputs, proprioception)
        _, outputs = torch.max(outputs_probas.data, 1)

        probas_y.extend(outputs_probas.data.cpu().numpy().tolist())
        test_y.extend(labels.data.cpu().numpy().flatten().tolist())

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
    tpr =  ( tp / p ) if p != 0 else 0
    print("sensitivity - true positive rate (TPR) = {}".format(tpr))

    #When it is actually env, how often does it predicted self.
    #predicted as self, but actually it is env
    fpr = ( fp / n ) if n != 0 else 0
    print("false alarm - False positive rate (FPR) = {}".format(fpr))

    #When it is actually env, how often does it predicted env (TNR).
    #Specificity
    tnr = tn / n
    print("Specificity - true nigative rate (TNR) = {}".format(tnr))

    #when predict self, how often it is correct.
    #Positive predictive value (PPV), precision
    ppv = tp / (tp+fp) if (tp+fp) != 0 else 0
    print("precision - Positive predictive value (PPV) = {}".format(ppv))

    #confusionMatrix(cm, accuracy, total)
    plt_roc(test_y, probas_y, prefix)


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

def add_prefix(prefix, path):
    return os.path.join(prefix, path)

def plt_roc(test_y, probas_y, prefix, plot_micro=False, plot_macro=False):
    assert isinstance(test_y, list) and isinstance(probas_y, list), 'the type of input must be list'
    skplt.metrics.plot_roc(test_y, probas_y, plot_micro=plot_micro, plot_macro=plot_macro)
    plt.savefig(add_prefix(prefix, 'roc_auc_curve.png'))
    plt.close()

class arch3model(nn.Module):
    def __init__(self):
        super(arch3model, self).__init__()
        self.model_ft = models.resnet18(pretrained=True)
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, 19)
        self.model_ft = self.model_ft.to(device)
        
        self.fc1 = nn.Linear(70, 50)
        self.fc2 = nn.Linear(50, 2)
        
    def forward(self, image, proprioception):
        outof_resnet18 = self.model_ft(image) #outof_resnet18.size() torch.Size([64, 19])
        #pro = proprioception
        
        x = torch.cat((outof_resnet18, proprioception), dim=1)
        x = F.relu(self.fc1(x)) #x.size() torch.Size([64, 70])
        x = self.fc2(x)
        return x
        

model_arc3 = arch3model()
model_arc3 = model_arc3.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(model_arc3.parameters(), lr=0.001, momentum=0.9)

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model_arc3.state_dict():
    print(param_tensor, "\t", model_arc3.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer_ft.state_dict():
    print(var_name, "\t", optimizer_ft.state_dict()[var_name])

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
train_mode = False
if train_mode == True:
    model_arc3 = train_model(model_arc3, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)
    torch.save(model_arc3.state_dict(), "model_arc3_save.pth")

state_dict = torch.load("model_arc3_save.pth") 
model_arc3.load_state_dict(state_dict)
#model_arc3 = eval_model(model_arc3, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)

#visualize_model(model_arc3)
# have the confusion matrix.
accuracy(model_arc3)

#plt.ioff()
#plt.show()