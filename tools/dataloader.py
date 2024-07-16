import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

def readImg(path):
    return Image.open(path)

def normalize(x, a, b, c, d):
    '''
    x from (a, b) to (c, d)
    '''
    return (float(x) - a) * (float(d) - c) / (float(b) - a) + float(c)

class MyDataset(Dataset):
    def __init__(self, img_path, data_transforms=None, loader=readImg, is_train=True, img_size=256):
        self.data_transforms = data_transforms
        self.loader = loader
        self.img_size=img_size
        self.is_train=is_train

        if is_train == True:
            img_path = '{}/train'.format(img_path)
        elif is_train == False:
            img_path = '{}/val'.format(img_path)

        # GT
        self.gt_file_name = '{}/GT/*'.format(img_path)
        gt_file = glob.glob(self.gt_file_name)
        gt_file.sort()
        self.gt_img = []
        for idx in range(len(gt_file)):
            tmp = glob.glob('{}/*.PNG'.format(gt_file[idx]))
            tmp.sort()
            self.gt_img.extend(tmp.copy())

        # NOISY
        self.noisy_file_name = '{}/NOISY/*'.format(img_path)
        noisy_file = glob.glob(self.noisy_file_name)
        noisy_file.sort()
        self.noisy_img = []
        for idx in range(len(noisy_file)):
            tmp = glob.glob('{}/*.PNG'.format(noisy_file[idx]))
            tmp.sort()
            self.noisy_img.extend(tmp.copy())
        
        # RED
        self.red_file_name = '{}/RED/*'.format(img_path)
        red_file = glob.glob(self.red_file_name)
        red_file.sort()
        self.red_img = []
        for idx in range(len(red_file)):
            tmp = glob.glob('{}/*.PNG'.format(red_file[idx]))
            tmp.sort()
            self.red_img.extend(tmp.copy())

        # PARAM
        self.param_file_name = '{}/PARAM/*'.format(img_path)
        param_file = glob.glob(self.param_file_name)
        param_file.sort()
        self.param = []
        for idx in range(len(param_file)):
            tmp = glob.glob('{}/*.txt'.format(param_file[idx]))
            tmp.sort()
            self.param.extend(tmp.copy())


        self.red_ = []
        self.param_ = []
     
        
        for idx in tqdm(range(len(self.gt_img))):
             
             red_name = self.red_img[idx]
             param_name = self.param[idx]
            
             red_ = self.loader(red_name)

             lines_ = []
             param_ = []
             #param_denorm = [] 
             with open(param_name, 'r') as f:
                 line = f.readline()
                 while line:
                     lines_.append(line)
                     line = f.readline()
             f.close()
             param_.append(normalize(float(lines_[0]), 1, 15, 0, 1))
             param_.append(normalize(float(lines_[1]), 4, 8, 0, 1))
             param_.append(0. if lines_[2][:3]=='opp' else 1.)
             param_.append(0. if lines_[3][:3]=='dct' else 1.)
             param_.append(normalize(float(lines_[4]), 4, 15, 0, 1))

             
             
             if self.data_transforms is not None:
                 if self.is_train == True:
                     x, y, w, h = transforms.RandomCrop.get_params(red_, (self.img_size, self.img_size))

                     red_ = self.data_transforms(red_.crop([x, y, x+w, y+h]))
                 else:

                     red_ = self.data_transforms(red_)

             param_ = torch.tensor(param_)
             
             self.red_.append(red_.unsqueeze(0))
             self.param_.append(param_.unsqueeze(0))
             
        self.red_ = torch.cat(self.red_, 0)
        self.param_ = torch.cat(self.param_, 0)
        


    def __len__(self):
        return len(self.param_)

    def __getitem__(self, item):

        gt_name = self.gt_img[item]
        noisy_name = self.noisy_img[item]
        gt_ = self.loader(gt_name)
        noisy_ = self.loader(noisy_name)   

        if self.data_transforms is not None:
            if self.is_train == True:
                x, y, w, h = transforms.RandomCrop.get_params(gt_, (self.img_size, self.img_size))
                gt_ = self.data_transforms(gt_.crop([x, y, x+w, y+h]))
                noisy_ = self.data_transforms(noisy_.crop([x, y, x+w, y+h]))
                
            else:
                gt_ = self.data_transforms(gt_)
                noisy_ = self.data_transforms(noisy_)
                

        
        return gt_, noisy_, self.red_[item], self.param_[item]

    
    def update_data(self, index, red, cff, bs_ht, cspace, transform_2d_wiener_name, bs_wiener):
        self.red_[index].copy_(red)
        temp_param = [cff, bs_ht, cspace, transform_2d_wiener_name, bs_wiener]
        temp_param = torch.tensor(temp_param)
        self.param_[index].copy_(temp_param)

     



class MyDatasets2(Dataset):
    def __init__(self, img_path, data_transforms=None, loader=readImg, is_train=True, img_size=256):
        self.data_transforms = data_transforms
        self.loader = loader
        self.img_size=img_size
        self.is_train=is_train

        if is_train == True:
            img_path = '{}/train'.format(img_path)
        elif is_train == False:
            img_path = '{}/val'.format(img_path)

        # GT
        self.gt_file_name = '{}/GT/*'.format(img_path)
        gt_file = glob.glob(self.gt_file_name)
        gt_file.sort()
        self.gt_img = []
        for idx in range(len(gt_file)):
            tmp = glob.glob('{}/*.PNG'.format(gt_file[idx]))
            tmp.sort()
            self.gt_img.extend(tmp.copy())

        # NOISY
        self.noisy_file_name = '{}/NOISY/*'.format(img_path)
        noisy_file = glob.glob(self.noisy_file_name)
        noisy_file.sort()
        self.noisy_img = []
        for idx in range(len(noisy_file)):
            tmp = glob.glob('{}/*.PNG'.format(noisy_file[idx]))
            tmp.sort()
            self.noisy_img.extend(tmp.copy())
        
        # RED
        self.red_file_name = '{}/RED/*'.format(img_path)
        red_file = glob.glob(self.red_file_name)
        red_file.sort()
        self.red_img = []
        for idx in range(len(red_file)):
            tmp = glob.glob('{}/*.PNG'.format(red_file[idx]))
            tmp.sort()
            self.red_img.extend(tmp.copy())

        # PARAM
        self.param_file_name = '{}/PARAM/*'.format(img_path)
        param_file = glob.glob(self.param_file_name)
        param_file.sort()
        self.param = []
        for idx in range(len(param_file)):
            tmp = glob.glob('{}/*.txt'.format(param_file[idx]))
            tmp.sort()
            self.param.extend(tmp.copy())

    def __len__(self):
        return len(self.param)

    def __getitem__(self, item):
        gt_name = self.gt_img[item]
        noisy_name = self.noisy_img[item]
        red_name = self.red_img[item]
        param_name = self.param[item]

        gt_ = self.loader(gt_name)
        noisy_ = self.loader(noisy_name)
        red_ = self.loader(red_name)

        lines_ = []
        param_ = []

        with open(param_name, 'r') as f:
            line = f.readline()
            while line:
                lines_.append(line)
                line = f.readline()
        f.close()
        param_.append(normalize(float(lines_[0]), 1, 15, 0, 1))
        param_.append(normalize(float(lines_[1]), 4, 8, 0, 1))
        param_.append(0. if lines_[2][:3]=='opp' else 1.)
        param_.append(0. if lines_[3][:3]=='dct' else 1.)
        param_.append(normalize(float(lines_[4]), 4, 15, 0, 1))

        if self.data_transforms is not None:
            if self.is_train == True:
                x, y, w, h = transforms.RandomCrop.get_params(gt_, (self.img_size, self.img_size))
                gt_ = self.data_transforms(gt_.crop([x, y, x+w, y+h]))
                noisy_ = self.data_transforms(noisy_.crop([x, y, x+w, y+h]))
                red_ = self.data_transforms(red_.crop([x, y, x+w, y+h]))
            else:
                gt_ = self.data_transforms(gt_)
                noisy_ = self.data_transforms(noisy_)
                red_ = self.data_transforms(red_)
        
        return gt_, noisy_, red_, torch.tensor(param_)

def train_dataloader(img_path, batch_size=16, num_threads=4, shuffle=False, img_size=256):
    transform = transforms.Compose([
        
        transforms.ToTensor(),
    
    ])
    dataset = MyDatasets2(img_path=img_path, data_transforms=transform, is_train=True, img_size=img_size)


    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                                                num_workers=num_threads, pin_memory=True)
    return dataloader

def val_dataloader(img_path, batch_size=16, num_threads=4, shuffle=False, img_size=256):
    transform = transforms.Compose([
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
       
    ])
    dataset = MyDatasets2(img_path=img_path, data_transforms=transform, is_train=False, img_size=img_size)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                                                num_workers=num_threads, pin_memory=True)
    return dataloader




class MyDataset2(Dataset):
    def __init__(self, img_path, data_transforms=None, loader=readImg, img_size=256):
        self.data_transforms = data_transforms
        self.loader = loader
        self.img_size=img_size

        img_path = '{}/test'.format(img_path)
        print(img_path)

        # GT
        self.gt_file_name = '{}/GT/*'.format(img_path)
        gt_file = glob.glob(self.gt_file_name)
        gt_file.sort()
        #print(gt_file)
        self.gt_img = []
        for idx in range(len(gt_file)):
            tmp = glob.glob('{}/*.PNG'.format(gt_file[idx]))
            tmp.sort()
            self.gt_img.extend(tmp.copy())
        
        # NOISY
        self.noisy_file_name = '{}/NOISY/*'.format(img_path)
        noisy_file = glob.glob(self.noisy_file_name)
        noisy_file.sort()
        self.noisy_img = []
        for idx in range(len(noisy_file)):
            tmp = glob.glob('{}/*.PNG'.format(noisy_file[idx]))
            tmp.sort()
            self.noisy_img.extend(tmp.copy())
        

    def __len__(self):
        return len(self.gt_img)

    def __getitem__(self, item):
        gt_name = self.gt_img[item]
        noisy_name = self.noisy_img[item]

        gt_ = self.loader(gt_name)
        noisy_ = self.loader(noisy_name)

        if self.data_transforms is not None:
            gt_ = self.data_transforms(gt_)
            noisy_ = self.data_transforms(noisy_)

        return gt_, noisy_

def test_dataloader(img_path, batch_size=1, num_threads=1, shuffle=False, img_size=256):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = MyDataset2(img_path=img_path, data_transforms=transform, img_size=img_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                                                num_workers=num_threads, pin_memory=True)
    return dataloader
