import torch
from torchvision import datasets
from torchvision import transforms
import yaml

import os

dx = {}


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        pro = getproprioception(self, index, path)
        if pro is not None:
            # print(path)
            # print(type(pro))
            tensorpro = torch.tensor(pro)
            # print(tensorpro)
            # print(type(tensorpro))
            # print(transforms.ToTensor())
        return sample, target, path, tensorpro


def getfilename(self, index, path, proprioception):
    # print("the index {}, and path{} ".format(index, path))
    # with open("./dataset/pro{}_{}.txt".format(counter, 
    #           proprioseption.header.stamp.secs), 'r') as outputfile:
    # print(path)
    # maping the image with corresponding pro.
    # From: pathcopyfromboth_16-10-2018\train\baxter\images\image26106_2034.jpg
    # To  :pathcopyfromboth_16-10-2018\train\baxter\pro\pro26106_2034.txt
    file = path.replace("images", "pro")
    file = file.replace("image", "pro")
    file = file.replace(".jpg", ".txt")
    ######
    # used for one file with all proproception elements are randomized
    
    #point to the file path
    #open the file
    #get the values of the key of file variable name
    #if "train" not in (self.root) and "env" in path:
    allprotakenfromonefileisactive = True
    if "train" not in (self.root) and "env" in path and allprotakenfromonefileisactive:
        # print("validation time")
        # for validation only (val/env/pro/ not used)
        # take proprioception information from random file in /randomuncyned/allrandomwhichisuncynedpro.txt.
        global dx
        if len(dx) == 0:
            filespath = os.path.abspath(__package__)
            #randomproprioceptionfile = filespath + '/' + '20190415' + '/' + 'randomuncyned' + '/' + 'allrandomunsyncedprobasedonproofenvranges.txt'
            #randomproprioceptionfile = filespath + '/' + '20190416' + '/' + 'randomuncyned' + '/' + 'allrandomunsyncedprobasedonproofbaxterranges.txt'
            
            #randomproprioceptionfile = filespath + '/' + '20190429' + '/' + 'randomuncyned' + '/' + 'allrandomunsyncedprobasedonproofenvranges.txt'
            #randomproprioceptionfile = filespath + '/' + '20190429' + '/' + 'randomuncyned' + '/' + 'allrandomunsyncedprobasedonproofenvranges_of_training_data.txt'
            
            #randomproprioceptionfile = filespath + '/' + '20190514-case3' + '/' + 'envrange' + '/' + 'envproinformation_associatedwith_baxtervalpronames.txt'
            #randomproprioceptionfile = filespath + '/' + '20190514-case4' + '/' + 'baxterrange' + '/' + 'baxterproinformation_associatedwith_valenvpronames.txt'

            randomproprioceptionfile = filespath + '/' + '20190521-case3' + '/' + 'envrange' + '/' + 'envproinformation_associatedwith_baxtervalpronames.txt'
            #randomproprioceptionfile = filespath + '/' + '20190521-case4' + '/' + 'baxterrange' + '/' + 'baxterproinformation_associatedwith_valenvpronames.txt'
            with open(randomproprioceptionfile, 'r') as outputfile:
                try:
                    x = outputfile.read()
                except outputfile.errors as exc:
                    print(exc)
            dx = yaml.load(x)
        keyAsfilenameonly = os.path.basename(file)
        x = dx[keyAsfilenameonly] 
    ######
    else:
        with open(file, 'r') as outputfile:
            try:
                x = outputfile.read()
            except outputfile.errors as exc:
                print(exc)
    # print(type(x))
    # print("Yaml load the txt file--------------------")
        x = yaml.load(x)
    # print(x)
    # print(type(x))
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
    proprioception = proprioception['velocity'] +  proprioception['effort'] + proprioception['position']
    return proprioception