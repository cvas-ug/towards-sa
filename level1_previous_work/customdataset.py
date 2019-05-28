import torch
from torchvision import datasets
from torchvision import transforms
import yaml


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
        pro = getproprioceptions(index, path)
        if pro is not None:
            # print(path)
            # print(type(pro))
            tensorpro = torch.tensor(pro)
            # print(tensorpro)
            # print(type(tensorpro))
            # print(transforms.ToTensor())
        return sample, target, path, tensorpro


def getproprioceptions(index, path):
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
    with open(file, 'r') as outputfile:
        try:
            x = outputfile.read()
        except outputfile.errors as exc:
            print(exc)
    # print(type(x))
    # print("Yaml load the txt file--------------------")
    x = yaml.load(x)
    return x["velocity"]
    #return x["effort"]
    # print(x)
    # print(type(x))                
