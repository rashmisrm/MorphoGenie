
import torch
import torch.utils.data
import torch.utils.data.distributed
#import torchvision


def save_images(imgs, outfile, nrow=8):
    imgs = imgs / 2 + 0.5     # unnormalize
    torchvision.utils.save_image(imgs, outfile, nrow=nrow)


def get_nsamples(data_loader, N):
    x = []
    y = []
    n = 0
    while n < N:
        


        #if __name__=='__main__':
        #x_next, y_next = next(iter(data_loader))
        x_next, y_next, z_next = next(iter(data_loader))
        x.append(x_next)
        y.append(y_next)
        n += x_next.size(0)
        x = torch.cat(x, dim=0)[:N]
        #y = torch.cat(y, dim=0)[:N]
        return x, y
    



def update_average(model_tgt, model_src, beta):
    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert(p_src is not p_tgt)
        p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)


"""dataset.py"""

import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms


def is_power_of_2(num):
    return ((num & (num - 1)) == 0) and num != 0


class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)
        self.indices = range(len(self))

    def __getitem__(self, index1):
        index2 = random.choice(self.indices)

        path1 = self.imgs[index1][0]
        #path2 = self.imgs[index2][0]
        img1 = self.loader(path1)
        #img2 = self.loader(path2)
        if self.transform is not None:
            img1 = self.transform(img1)
            #img2 = self.transform(img2)
            
        return img1#, img2


class CustomImageFolderTest(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolderTest, self).__init__(root, transform)
        self.indices = range(len(self))

    def __getitem__(self, index1):
        index2 = random.choice(self.indices)

        path1 = self.imgs[index1][0]
        img1 = self.loader(path1)
        name= os.path.split(path1)
        name1=os.path.split(name[0])
        name2=os.path.split(name1[0])


        Label1=name1[1]
        Label2=name2[1]
       # img2 = self.loader(path2)
        if self.transform is not None:
            img1 = self.transform(img1)
            
        return img1, path1, Label2
    


class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor, transform=None):
        self.data_tensor = data_tensor
        self.transform = transform
        self.indices = range(len(self))

    def __getitem__(self, index1):
        index2 = random.choice(self.indices)

        img1 = self.data_tensor[index1]
        img2 = self.data_tensor[index2]
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1

    def __len__(self):
        return self.data_tensor.size(0)

class CustomImageFolderCombine3(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolderCombine3, self).__init__(root,transform)
        self.indices = range(len(self.imgs))

    def __getitem__(self, i11):
        i13 = i12 = i11 

        i21 = random.choice(self.indices)
        i23 = i22 = i21
        path11 = self.imgs[i11][0]
        name= os.path.split(path11)[0]
        name= os.path.split(name)[0]
        Type=os.path.split(name)[1]
        name= os.path.split(name)[0]
        name= os.path.split(name)[0]

        path12 = name+'\\Ch2\\'+ Type + '\\Ch\\'+ os.path.basename(path11)
        path13 = name+'\\Ch3\\'+ Type + '\\Ch\\'+ os.path.basename(path11)

        path21 = self.imgs[i21][0]
        
        name= os.path.split(path21)[0]
        name= os.path.split(name)[0]
        Type=os.path.split(name)[1]
        name= os.path.split(name)[0]
        name= os.path.split(name)[0]


        path22 = name+'\\Ch2\\'+ Type + '\\Ch\\'+ os.path.basename(path21)
        path23 = name+'\\Ch3\\'+ Type + '\\Ch\\'+ os.path.basename(path21)
        #path22 = '..\\data\\ProcessedData\\CPA\Treatment1\\Ch2\\'+os.path.basename(path21)
        #path23 = '..\\data\\ProcessedData\\CPA\Treatment1\\Ch3\\'+os.path.basename(path21)

        img1S2 = self.loader(path21)
        img2S2 = self.loader(path22)
        img3S2 = self.loader(path23)

        #print('rashmi')
        img1S1 = self.loader(path11)
        img2S1 = self.loader(path12)
        img3S1 = self.loader(path13)

        
        img1S2 = self.loader(path21)
        img2S2 = self.loader(path22)
        img3S2 = self.loader(path23)
        
        if self.transform is not None:
            img1S1 = self.transform(img1S1)
            img2S1 = self.transform(img2S1)
            img3S1 = self.transform(img3S1)
                        
            img1S2 = self.transform(img1S2)
            img2S2 = self.transform(img2S2)
            img3S2 = self.transform(img3S2)
        img1=torch.cat((img1S1, img2S1, img3S1),0)
        img2=torch.cat((img1S2, img2S2, img3S2),0)

        return img1
    
    
class CustomImageFolderCombine3Test(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolderCombine3Test, self).__init__(root,transform)
        self.indices = range(len(self.imgs))

    def __getitem__(self, i11):
        i13 = i12 = i11 

        i21 = random.choice(self.indices)
        i23 = i22 = i21
        path11 = self.imgs[i11][0]
        name= os.path.split(path11)[0]
        name= os.path.split(name)[0]
        Type=os.path.split(name)[1]
        name= os.path.split(name)[0]
        name= os.path.split(name)[0]

        path12 = name+'\\CH2\\'+ Type + '\\Ch\\'+ os.path.basename(path11)
        path13 = name+'\\CH3\\'+ Type + '\\Ch\\'+ os.path.basename(path11)


  
        #print('rashmi')
        img1S1 = self.loader(path11)
        img2S1 = self.loader(path12)
        img3S1 = self.loader(path13)

        if self.transform is not None:
            img1S1 = self.transform(img1S1)
            img2S1 = self.transform(img2S1)
            img3S1 = self.transform(img3S1)
            
            
        img1=torch.cat((img1S1, img2S1, img3S1),0)
        
        name= os.path.split(path11)
        name=os.path.split(name[0])
        name1=os.path.split(name[0])
        name2=os.path.split(name1[0])

        Label1=name1[1]
        Label2=name2[1]
        
            
        return img1, Label1, Label2    
def return_data(args):
    name = args.dataset
    imtype= args.datatype
    dset_dir = args.dset_dir
    
    batch_size = args.bs
    num_workers = args.num_workers
    image_size = args.image_size
    #assert image_size == 64, 'currently only image size of 64 is supported'


    if imtype.lower() == 'rgb':
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Lambda(lambda x: x + 1./128 * torch.rand(x.size()))])
    
        root = os.path.join(dset_dir, name)
        train_kwargs = {'root':root, 'transform':transform}
        dset = CustomImageFolder
        train_data = dset(**train_kwargs)
        
        
        train_loader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  drop_last=True)
        data_loader = train_loader
        return data_loader
    elif imtype.lower() == 'gray': 
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5)),
                transforms.Lambda(lambda x: x + 1./128 * torch.rand(x.size()))])
        
            root = os.path.join(dset_dir, name)
            train_kwargs = {'root':root, 'transform':transform}
            dset = CustomImageFolder
            train_data = dset(**train_kwargs)
            train_loader = DataLoader(train_data,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      num_workers=num_workers,
                                      pin_memory=True,
                                      drop_last=True)
            data_loader = train_loader
            return data_loader
    elif imtype.lower() == 'cpa': 
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
            transforms.Lambda(lambda x: x + 1./128 * torch.rand(x.size()))])
    
        root = os.path.join(dset_dir, name, 'CH1')
        train_kwargs = {'root':root, 'transform':transform}
        dset = CustomImageFolderCombine3
        train_data = dset(**train_kwargs)
        train_loader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  drop_last=True)
        data_loader = train_loader
        
        return data_loader


        #train_features, train_labels = next(iter(data_loader))
       # i=0
       # for x_true1, x_true2, path in data_loader:
       #     print(x_true1)
       #     i=i+1
       #     print(i)
       #     print(path)
        
'''    elif imtype.lower() == 'dicxy':
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(),
            transforms.ToTensor()])

        root = os.path.join(dset_dir, 'DICxy/G2/x')

        train_kwargs = {'root':root,'transform':transform}
        dset = CustomImageFolderCombineTest
        
        train_data = dset(**train_kwargs)
        train_loader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  drop_last=True)
        data_loader = train_loader
        


        i=0

    else:
        raise NotImplementedError'''


def return_data_test(args):
    name = args.dataset
    imtype= args.datatype
    dset_dir = args.dset_dir
    
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = args.image_size
    #assert image_size == 64, 'currently only image size of 64 is supported'

    if imtype.lower() == 'rgb':
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Lambda(lambda x: x + 1./128 * torch.rand(x.size()))])
    
        root = os.path.join(dset_dir, name)
        train_kwargs = {'root':root, 'transform':transform}
        dset = CustomImageFolderTest
        train_data = dset(**train_kwargs)
        
        
        train_loader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  drop_last=True)
        data_loader = train_loader
        return data_loader

    elif imtype.lower() == 'gray':
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5)),
                transforms.Lambda(lambda x: x + 1./128 * torch.rand(x.size()))])
        
            root = os.path.join(dset_dir, name)
            train_kwargs = {'root':root, 'transform':transform}
            dset = CustomImageFolderTest
            train_data = dset(**train_kwargs)
            train_loader = DataLoader(train_data,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      num_workers=num_workers,
                                      pin_memory=True,
                                      drop_last=True)
            data_loader = train_loader
            return data_loader

        
    elif imtype.lower() == 'cpa':
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
            transforms.Lambda(lambda x: x + 1./128 * torch.rand(x.size()))])
    
        root = os.path.join(dset_dir, name, 'CH1')
        train_kwargs = {'root':root, 'transform':transform}
        dset = CustomImageFolderCombine3Test
        train_data = dset(**train_kwargs)
        train_loader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  drop_last=True)
        data_loader = train_loader
        return data_loader

