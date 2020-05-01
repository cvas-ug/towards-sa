# ploting inspired with explination from https://youtu.be/sE8kwierv_4?t=671
import torch
from torchvision import transforms
import os
import customdataset

import torchvision
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import itertools
import time

plt.ion()

data_transforms = {
    'train': transforms.Compose([
        #transforms.RandomResizedCrop(224),
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        #transforms.Resize(256),
        #transforms.CenterCrop(224),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


data_dir = "copyfromboth_16-10-2018"
#data_dir = '20190220-21'
# data_dir = os.path.dirname(__file__) + "/" + data_dir

image_datasets = {x: customdataset.ImageFolderWithPaths(
                                   os.path.join(data_dir, x),
                                   data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                              shuffle=True, num_workers=0)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# device = torch.device("cpu")

#torch.cuda.manual_seed(2)
#torch.manual_seed(2)

torch.cuda.manual_seed(24)
torch.manual_seed(24)

#torch.cuda.manual_seed(2481)
#torch.manual_seed(2481)

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
    plt.pause(5)

# Visualizing the model predictions
# Generic function to display predictions for a few images
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels, path, proprioception) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            proprioception = proprioception.to(device)

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


class Netz(nn.Module):
    def __init__(self):
        super(Netz, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=5)
        self.conv3 = nn.Conv2d(12, 18, kernel_size=5)
        self.conv4 = nn.Conv2d(18, 24, kernel_size=5)
        # self.fc1 = nn.Linear(18 * 28 * 28, 1000)
        self.fc1 = nn.Linear(3360, 1000) #2400, 1000)
        self.fc2 = nn.Linear(1019, 2)
        self.fc3 = nn.Linear(19, 19)

    def forward(self, x, pro):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # use this to calculate the view size
        #print(x.size())
        #exit()
        # x = x.view(-1, 18 * 28 * 28)
        #x = x.view(-1, 3456)
        x = x.view(-1, 3360) #2400)
        x = F.relu(self.fc1(x))
    
        p = self.fc3(pro)
        
        # now we can reshape `c` and `f` to 2D and concat them
        combined = torch.cat((x.view(x.size(0), -1), p.view(p.size(0), -1)), dim=1)

        out = self.fc2(combined)
        # print(x.size())

        # return F.sigmoid(x) 
        return out
        #return F.log_softmax(x)


def train(num_epochs=32):
    model.train()
    closs_average = []
    for epoch in range(num_epochs):
        batch_id = 0
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        losses = []
        closs = 0
        
        # If I got cuda
        # data = data.cuda()
        # target = torch.Tensor(target).cuda()
        # target = torch.LongTensor(target).cuda()
        
        # Iterate over data.
        for i, batch in enumerate(dataloaders['train'], 0):
            batch_id += 1
            inputs, labels, path, proprioception = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            proprioception = proprioception.to(device)

            optimizer.zero_grad()
        
            # forward
            # track history if only in train
            outputs = model(inputs, proprioception)
            criterion = nn.CrossEntropyLoss()
            #_, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            closs += loss.item()
            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()

            #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #    epoch, batch_id * len(inputs), len(dataloaders['train']),
            #        100. * batch_id / len(dataloaders['train']), loss.data[0]))
            #batch_id = batch_id + 1

            # track every 100 loss    
            if i % 100 == 0:
                losses.append(loss.item())
                print('this Loss: {:.6f}'.format(loss.item()))

            if ((i+1) % dataset_sizes['train'] == 0) and ((i+1) != 0):
                print('[%d   %d] loss: %.6f' % (epoch+1, i+1, closs/dataset_sizes['train']))
                closs_average.append(closs/dataset_sizes['train'])
                closs = 0
        # Calculate the accuracy and save the model state
        #accuracy()
        # Plot the graph of loss with iteration
        #plt.xlim(0, batch_id)
        
    #     t1 = np.arange(0.0, 1500, 100)
    #     plt.plot(t1, losses, label='epoch'+str(epoch))
    #     plt.legend(loc='upper left', mode="expanded", shadow=True, ncol=2)
    # plt.xlabel("Batch number : (plot every 100 from batch)")
    # plt.ylabel("Loss")
    # plt.title("Arch2 training losses")
    # plt.show()
    t1 = np.arange(0, num_epochs, 1)
    plt.plot(t1, closs_average)# ,label='epoch'+str(epoch))
    #plt.legend(loc='upper left', mode="expanded", shadow=True, ncol=2)
    plt.xlabel("Epoch number", fontsize=12, color='blue')
    plt.ylabel("Average loss", fontsize=12, color='blue')
    plt.title("Arch2 training losses")
    print("---")
    print(closs_average)
    plt.show()


def confusionMatrix(cm, accuracy, total):
    target_names = ['Self', 'Environment']
    title = "Confusion Matrix" + ':Accuracy ={} on total of batch {}'.format(accuracy, total)
    cmap = "Greens"

    if cmap is None:
        cmap = plt.get_cmap("Blues")

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
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
    plt.show(5)


def accuracy():
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

        outputs = model(inputs, proprioception)
        _, outputs = torch.max(outputs.data, 1)  # return max as well as its index
        total += labels.size(0)
        correctHits += (outputs == labels).sum().item()
        accuracy = (correctHits/total)*100
        val = [outputs.item(), labels.item()]
        print(val)
        if val == [0, 0]:
            trueSelf_predictedSelf += 1
        if val == [0, 1]:
            trueSelf_predictedEnvironment += 1
        if val == [1, 0]:
            trueEnvironment_predictedSelf += 1
        if val == [1, 1]:
            trueEnvironment_predictedEnvironment += 1
    print('Accuracy ={} on total of batch {}'.format(accuracy, total))
    cm = np.array([[trueSelf_predictedSelf, trueSelf_predictedEnvironment],
                [trueEnvironment_predictedSelf, trueEnvironment_predictedEnvironment]])
    # From self prospective:
    # P condition positive: the number of real positive cases in the data (tp+)
    # N condition negative: the number of real negative cases in the data
    tp = trueSelf_predictedSelf
    fp = trueSelf_predictedEnvironment 
    fn = trueEnvironment_predictedSelf
    tn = trueEnvironment_predictedEnvironment
    print(trueSelf_predictedSelf + trueSelf_predictedEnvironment)
    print(trueEnvironment_predictedEnvironment + trueEnvironment_predictedSelf)
    print(tp)
    print(fp)
    print(fn)
    print(tn)
    #Accuracy (ACC) = Σ True positive + Σ True negative / Σ Total population
    acc = (tp + tn)/(tp+fp+fn+tn)
    print("acc = {}".format(acc))
    #Prevalence = Σ Condition positive / Σ Total population
    #prevalence = 

    #sensitivity, recall, hit rate, or true positive rate (TPR)
    #tpr = tp / p

    confusionMatrix(cm, accuracy, total)


if __name__ == '__main__':
    start = time.time()
    model = Netz()
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # Get a batch of training data
    # inputs, classes, path, pro = next(iter(dataloaders['train']))
    # print(path)
    train(num_epochs=2)
    end = time.time()
    print("The training only took about : " + str((end - start) / 60))
    visualize_model(model)
    accuracy()
    
    # Make a grid from batch    
    #out = torchvision.utils.make_grid(inputs)
    #imshow(out, title=[class_names[x] for x in classes])
