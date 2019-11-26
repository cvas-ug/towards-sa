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

import dataloading
import customdataset
import torch.nn.functional as F

import itertools
import scikitplot as skplt

from flashtorch.utils import apply_transforms, load_image
from flashtorch.saliency import Backprop
from flashtorch.activmax import GradientAscent
import pickle

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

#data_dir = '20190611'       #case1and2 real baxter data
#data_dir = '20190611-case3' #case3 - real baxter data
#data_dir = '20190611-case4'  #case4 - real baxter data

#data_dir = '20190612'       #case1and2 real baxter data including cases3and4 in the training data.
#data_dir = '20190612-case3' #case3 - real baxter data including cases3and4 in the training data.
#data_dir = '20190612-case4' #case4 - real baxter data including cases3and4 in the training data.

#data_dir = '20190612-test' #has only baxter just for test the visualisation

#image_datasets = {x: customdataset.ImageFolderWithPaths(os.path.join(data_dir, x),
#                                          data_transforms[x])
#                  for x in ['train', 'val']}

# activate/deactivate training
train_mode = False

# activate/deactivate testset
#test_group = None

##########################################################################
#
# The following is the configuration for four groups experiments FC, IL, FG, FT
# here the training was on case1(me) and case2(env)
#
##########################################################################

# load model state of group
#exprimentalgroup = "exp1ilfgft"
#exprimentalgroup = "exp2fcilfg"
#exprimentalgroup = "exp3fcfgft"
#exprimentalgroup = "exp4fcilft"

# unseen test group
#test_group = "20190925unseen/20190925fc/20190925fc.csv"
#test_group = "20190925unseen/20190925ft/20190925ft.csv"
#test_group = "20190925unseen/20190925il/20190925il.csv"
#test_group = "20190925unseen/20190925fg/20190925fg.csv"

# training/eval groups
#dataset_group = "ilfgft"
#dataset_group = "fcilfg"
#dataset_group = "fcfgft"
#dataset_group = "fcilft"

##########################################################################
#
# The following is the configuration for four groups experiments FC, IL, FG, FT
# here the training was on case1(me), case2(env), case3(env), and case4(env)
#
##########################################################################

# training/eval groups
dataset_group = "ilfgft_caseall"
#dataset_group = "fcilfg_caseall"
#dataset_group = "fcfgft_caseall"
#dataset_group = "fcilft_caseall"

# unseen test group with all cases
test_group = "20190925unseen/20190925fc/20190925fc_caseall.csv"
#test_group = "20190925unseen/20190925ft/20190925ft_caseall.csv"
#test_group = "20190925unseen/20190925il/20190925il_caseall.csv"
#test_group = "20190925unseen/20190925fg/20190925fg_caseall.csv"

# unseen test group with separate cases
#test_group = "20190925unseen/20190925fc/20190925fc_case1.csv"
#test_group = "20190925unseen/20190925fc/20190925fc_case2.csv"
#test_group = "20190925unseen/20190925fc/20190925fc_case3.csv"
#test_group = "20190925unseen/20190925fc/20190925fc_case4.csv" #%18

#test_group = "20190925unseen/20190925ft/20190925ft_case1.csv"
#test_group = "20190925unseen/20190925ft/20190925ft_case2.csv"
#test_group = "20190925unseen/20190925ft/20190925ft_case3.csv"
#test_group = "20190925unseen/20190925ft/20190925ft_case4.csv" #%98

#test_group = "20190925unseen/20190925il/20190925il_case1.csv"
#test_group = "20190925unseen/20190925il/20190925il_case2.csv"
#test_group = "20190925unseen/20190925il/20190925il_case3.csv"
#test_group = "20190925unseen/20190925il/20190925il_case4.csv" #%28

#test_group = "20190925unseen/20190925fg/20190925fg_case1.csv"
#test_group = "20190925unseen/20190925fg/20190925fg_case2.csv"
#test_group = "20190925unseen/20190925fg/20190925fg_case3.csv"
#test_group = "20190925unseen/20190925fg/20190925fg_case4.csv" #%89

#test_group = "saliency/case1/case1.csv"
#test_group = "saliency/case2/case2.csv"
#test_group = "saliency/case3/case3.csv"
#test_group = "saliency/case4/case4.csv"

exprimentalgroup = "expilfgft_caseall"
#exprimentalgroup = "expfcilfg_caseall"
#exprimentalgroup = "expfcfgft_caseall"
#exprimentalgroup = "expfcilft_caseall"


image_datasets = {x: dataloading.SADataset(x, test_group,
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                             shuffle=True, num_workers=0)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
#class_names = image_datasets['train'].classes

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
if test_group:
    print("dataset     : " + dataset_group + " tested with "+ test_group )
else:
    print("dataset     : " + dataset_group)


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
#inputs, classes, path, tensorpro = next(iter(dataloaders['train']))

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
                ax.set_title('predicted: {}, label: {}'.format(preds.cpu().data[j], labels.cpu().data[j]))#class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

# this method is limited for evaluation only using saved best training state.
# the "plotlossaverages_eval" used does not make sense as it plot same state for every epoch.
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
        #val = [outputs.item(), labels.item()]
        values = torch.stack([outputs, labels], -1)
        for val in values:
            val = list(val)
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
    tnr = ( tn / n ) if n != 0 else 0
    print("Specificity - true nigative rate (TNR) = {}".format(tnr))

    #when predict self, how often it is correct.
    #Positive predictive value (PPV), precision
    ppv = tp / (tp+fp) if (tp+fp) != 0 else 0
    print("precision - Positive predictive value (PPV) = {}".format(ppv))

    confusionMatrix(cm, accuracy, total)
    plt_roc(test_y, probas_y, prefix)


def confusionMatrix(cm, accuracy, total):
    target_names = ['Self', 'Environment']
    title = "Confusion Matrix Arch3" + ':Accuracy ={} on total of batch {}'.format(accuracy, total)
    cmap = "Greens"
    from matplotlib.ticker import MultipleLocator
    if cmap is None:
        cmap = plt.get_cmap("Blues")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=9, color='red')
    plt.colorbar()
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.set_xticklabels([''] + target_names)
    ax.set_yticklabels([''] + target_names, rotation=65)
    #if target_names is not None:
    #    tick_marks = np.arange(len(target_names))
    #    plt.xticks([''], target_names, rotation=45)
    #    plt.yticks([''], target_names)
    
    thresh = cm.max() / 1.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:,}".format(cm[i, j]),  ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label', fontsize=12, color='blue')
    plt.xlabel('Predicted label', fontsize=12, color='blue')
    plt.show()

def add_prefix(prefix, path):
    return os.path.join(prefix, path)

def plt_roc(test_y, probas_y, prefix, plot_micro=False, plot_macro=False):
    assert isinstance(test_y, list) and isinstance(probas_y, list), 'the type of input must be list'
    skplt.metrics.plot_roc(test_y, probas_y, plot_micro=plot_micro, plot_macro=plot_macro)
    plt.savefig(add_prefix(prefix, 'roc_auc_curve_'+ dataset_group +'.png'))
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

def visualise_max_gradient(testmodel):
    backprop = Backprop(testmodel)
    i=0
    for inputs, labels, path, tensorpro in dataloaders['val']:
        inputsfrompath = apply_transforms(load_image(path[0]))
        inputsfrompath =  inputsfrompath.to(device)
        inputs = inputs.to(device)
        inputs.requires_grad = True
        labels = labels.to(device)
        proprioception = tensorpro.to(device)
        proprioceptionge = proprioception
        proprioceptionge.requires_grad = True

        #class_index = class_names.index('baxter')
        class_index = labels

        #Calculate the gradients of each pixel w.r.t. the input image
        #the maximum of the gradients for each pixel across colour channels.
        backprop.visualize(inputs, proprioception, class_index, guided=True, use_gpu=False)
        print(labels)
        plt.ioff()
        i+=1
        print(i)
        dirctory = "saliency/case4/"
        case = "1"
        plt.savefig(dirctory+"train_"+exprimentalgroup+"_test"+case+str(i)+".png")
        #plt.show()
        #backprop.visualize(inputs, proprioceptionge, class_index, guided=True, use_gpu=True)
        #plt.ioff()
        #plt.show()
        #backprop.visualize(inputsfrompath, proprioception, class_index, guided=True, use_gpu=True)
        #plt.ioff()
        #plt.show()


def show_activation(testmodel):
    #list(testmodel.features.named_children())
    g_ascent =  GradientAscent(testmodel.model_ft)
    g_ascent.use_gpu = True

    conv1_1 = testmodel.model_ft.layer1[0].conv1
    conv1_1_filters = [17, 33, 34, 57]
    
    conv1_2 = testmodel.model_ft.layer1[0].conv2
    conv1_2_filters = [17, 33, 34, 57]

    conv2_1 = testmodel.model_ft.layer2[0].conv1
    conv2_1_filters = [17, 33, 34, 57]

    conv3_1 = testmodel.model_ft.layer3[0].conv1
    conv3_1_filters = [17, 33, 34, 57]

    conv4_2 = testmodel.model_ft.layer4[1].conv2
    conv4_2_filters = [7, 11, 63, 197, 203, 336, 438]



    #g_ascent.visualize(conv1_1, conv1_1_filters, title="conv1_1")
    #g_ascent.visualize(conv1_2, conv1_2_filters, title="conv1_2")
    #g_ascent.visualize(conv2_1, conv2_1_filters, title="conv2_1") 
    #g_ascent.visualize(conv4_2, conv4_2_filters, title="conv4_2") 
    for filter_no in range(0,511):
        output = g_ascent.visualize(conv4_2, filter_no, title=exprimentalgroup+"conv4_2", return_output=True)
        #print('num_iter:', len(output))
        #print('optimized image:', output[-1].shape)
        tensor_image = output[-1]
        tens = tensor_image
        torchvision.utils.save_image(tens, add_prefix("filters/"+exprimentalgroup, exprimentalgroup+"filter"+str(filter_no)+".png"), normalize=True)
    
    print("Generate ")

def get_module_weights(exprimentalgroup, layer):
    #current_model = arch3model()
    #current_model = current_model.to(device)

    #criterion = nn.CrossEntropyLoss()
    #optimizer_ft = optim.SGD(model_arc3.parameters(), lr=0.001, momentum=0.9)
    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    state_dict = torch.load("modelstate/model_arc3_save_20190925_"+ exprimentalgroup +".pth")
    print("load state : modelstate/model_arc3_save_20190925_"+ exprimentalgroup +".pth")
    loadedmodel = arch3model()
    loadedmodel.load_state_dict(state_dict)
    loadedmodel.eval()
    print(device)
    loadedmodel.to("cpu")

    #activations = {}
    #def get_activation(name):
    #    def hook(model, input, output):
    #        activations[name] = output.detach()
    #    return hook
    
    model = loadedmodel
    if layer == "fc2":
        weights = model.fc2.weight.data.numpy() #model[0].weight.data.numpy()
    if layer == "fc0":
        weights = model.model_ft.fc.weight.data.numpy()
    if layer == "fc1":
        weights = model.fc1.weight.data.numpy()

    print(weights.shape)
    #model[0].register_forward_hook(get_activation('layer0_relu'))
    #torch.manual_seed(7)
    #x = torch.randn(1, 10)
    #output = model(x)

    #plt.matshow(activations['layer0_relu'])
    #print(weights)
    #plt.matshow(weights)

    #plt.ioff()
    #plt.show()

    return weights

def calculate_mutual_information(weights):
    plt.rcParams['image.cmap'] = 'gray'
    plt.rcParams['image.interpolation'] = 'nearest'
    
    def mutual_info(w1, w2):
        # fig, axes = plt.subplots(1, 2)
        # axes[0].hist(w1.ravel(), bins=20)
        # axes[0].set_title('T1 slice histogram')
        # axes[1].hist(w2.ravel(), bins=20)
        # axes[1].set_title('T2 slice histogram')

        # plt.plot(w1.ravel(), w2.ravel(), '.')
        # plt.xlabel('T1 signal')
        # plt.ylabel('T2 signal')
        # plt.title('T1 vs T2 signal')
        # np.corrcoef(w1.ravel(), w2.ravel())[0, 1]

        # t1_20_30 = (w1 >= 0.0) & (w1 <= 0.30)
        # fig, axes = plt.subplots(1, 3, figsize=(8, 3))
        # axes[0].imshow(w1)
        # axes[0].set_title('T1 slice')
        # axes[1].imshow(t1_20_30)
        # axes[1].set_title('20<=T1<=30')
        # axes[2].imshow(w2)
        # axes[2].set_title('T2 slice')

        hist_2d, x_edges, y_edges = np.histogram2d(w1.ravel(), w2.ravel(), bins=20)
        plt.imshow(hist_2d.T, origin='lower')
        plt.xlabel('T1 signal bin')
        plt.ylabel('T2 signal bin')

        # hist_2d_log = np.zeros(hist_2d.shape)
        # non_zeros = hist_2d != 0
        # hist_2d_log[non_zeros] = np.log(hist_2d[non_zeros])
        # plt.imshow(hist_2d_log.T, origin='lower')
        # plt.xlabel('T1 signal bin')
        # plt.ylabel('T2 signal bin')

        # Mutual information for joint histogram
        # Convert bins counts to probability values
        pxy = hist_2d / float(np.sum(hist_2d))
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)
        px_py = px[:, None] * py[None, :]
        
        nzs = pxy > 0 
        return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

    
    count = 0
    mutual_values = []
    for w1 in weights:
        for w2 in weights:
            mutual_values.append(mutual_info(w1, w2))
            count+=1
    return mutual_values
    
def plot_table(weights):
    fig, axs =plt.subplots(2,1)
    clust_data = (np.array(weights)).reshape(4,4)
    collabel=("Group1", "Group2", "Group3", "Group4")
    rowslabel=("Group1", "Group2", "Group3", "Group4")
    axs[0].axis('tight')
    axs[0].axis('off')
    axs[1].axis('off')
    axs[0].table(cellText=clust_data,rowLabels=rowslabel, colLabels=collabel,loc='center')
    plt.show()

if __name__ == "__main__":   
    model_arc3 = arch3model()
    model_arc3 = model_arc3.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model_arc3.parameters(), lr=0.001, momentum=0.9)
    """
    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model_arc3.state_dict():
        print(param_tensor, "\t", model_arc3.state_dict()[param_tensor].size())

        # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer_ft.state_dict():
        print(var_name, "\t", optimizer_ft.state_dict()[var_name])
    """
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    if train_mode == True:
        model_arc3 = train_model(model_arc3, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)
        torch.save(model_arc3.state_dict(), "modelstate/model_arc3_save_20190925_exp"+dataset_group+".pth")
    else:
        state_dict = torch.load("modelstate/model_arc3_save_20190925_"+ exprimentalgroup +".pth")
        print("load state : modelstate/model_arc3_save_20190925_"+ exprimentalgroup +".pth")
        testmodel = arch3model()
        testmodel.load_state_dict(state_dict)
        testmodel.eval()
        print(device)
        testmodel.to(device)
        

    # show images with their predictions
    #visualize_model(testmodel)

    # generate confusion matrix and ROC.
    #accuracy(testmodel)

    # visualise using flashtorch (saliency maps)
    # works only with "batch_size=1 and num_workers=0"
    #visualise_max_gradient(testmodel)

    # activation maximization, get a patterns
    #show_activation(testmodel)

    # get weights of saved states
    exprimentalgroups = ["expilfgft_caseall", "expfcilfg_caseall", "expfcfgft_caseall", "expfcilft_caseall"]
    all_weights = []
    for exprimentgroup in exprimentalgroups:
        weights = get_module_weights(exprimentgroup, layer="fc2")
        all_weights.append(weights)
    #with open("fc0_all_group_weights.pkl", "wb") as p:
    #    pickle.dump(all_weights, p)
    #with open("fc0_all_group_weights.pkl", "rb") as p:
    #    ww = pickle.load(p)
    # calculate mutual information, and plot mutul info table
    mutual_info = calculate_mutual_information(all_weights)
    plot_table(mutual_info)

    
    plt.ioff()
    plt.show()