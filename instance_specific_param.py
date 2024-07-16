from unet_r import UNETR
from unet import UNet
import numpy as np
import torch
from tools.dataloader import train_dataloader,  val_dataloader, test_dataloader, MyDataset
from tqdm import tqdm
import visdom
from matplotlib import pyplot as plt
from PIL import Image
from tools.utils import get_psnr
import pickle
import os
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio
import torchvision.transforms as transforms
import torch.utils.data
import torchvision.datasets as datasets

# Proximal
import sys
sys.path.append('./ProxImaL')
from proximal.utils.utils import *
from proximal.utils.metrics import *
from proximal.lin_ops import *
from proximal.prox_fns import *
import cvxpy as cvx
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import glob
import random

# bm3d
sys.path.append('./bm3d-3.0.6')
from bm3d import bm3d_rgb
from bm3d.profiles import BM3DProfile
from experiment_funcs import get_psnr
from scipy.ndimage import correlate
from PIL import Image
import matplotlib.pyplot as plt
import glob
import os
from tqdm import tqdm





random.seed(1)

train_loss_step1_list = []
train_loss_step2_list = []
step2_epoch = 0 
train_epoch_step1_list = []
train_epoch_step2_list = []

step1_red_psnr_list = []
step1_net_psnr_list = []

step1_psnr_gt_red_list_test_mean = []
step1_psnr_gt_net_list_test_mean = []


step2_red_psnr_list = []
step2_net_psnr_list = []


step2_psnr_gt_red_list_test_mean = []
step2_psnr_gt_net_list_test_mean = []


def estimate_the_noise(img):
    I = np.asfortranarray((img))
    #print(I)
    I = np.mean(I, axis=2)
    I = np.asfortranarray(I)
    I = np.maximum(I, 0.0)
    ndev = estimate_std(I, 'daub_replicate')
    return ndev


def denormalize(y, a, b, c, d):
    '''
    y from (c, d) to (a, b)
    '''
    return (float(y) - c) * (float(b) - a) / (float(d) - c) + float(a)

def normalize(x, a, b, c, d):
    '''
    x from (a, b) to (c, d)
    '''
    return (float(x) - a) * (float(d) - c) / (float(b) - a) + float(c)

def train():
    #Approximation (Step 1) Initializtion
    MAX_EPOCH = 120
    LR = 0.005
    batch_size = 1
    
    print("Loading Data")
    
    image_dir = './SIDD_crop_bm3d'
    
    transform = transforms.Compose([  
        transforms.ToTensor(),
        
    ])
    dataset = MyDataset(img_path=image_dir, data_transforms=transform, is_train=True, img_size=512)
    
    train_loader_step1 = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                                                num_workers=16, pin_memory=True)
    
    
    val_loader_step1 = val_dataloader(image_dir, batch_size=1, num_threads=16, img_size=512)
    
    

    LR_step2 = 0.02
    

    #Step 1 :Model Initialization
    net_step1 = UNETR(in_channels = 3, out_channels = 3, img_size = (512,512), step_flag=1)
    net_step1 = torch.nn.DataParallel(net_step1).cuda()

    criterion_step1 = torch.nn.L1Loss()
    
    optimizer_step1 = torch.optim.Adam(net_step1.parameters(), LR)
    
    scheduler_step1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_step1, 20, eta_min=0, last_epoch=-1)
    profile = BM3DProfile()
    

    resume_epoch = 0
        
    if not os.path.exists('./checkpoints_step1'):
        os.mkdir('./checkpoints_step1')
    


    #Step 2: Model Initialization
    
    
    #Initialize Unet
    param_unet = UNet(in_channels=3, out_channels=5)
    param_unet = torch.nn.DataParallel(param_unet).cuda()
    
    criterion_step2 = torch.nn.MSELoss()
    optimizer_step2 = torch.optim.Adam(param_unet.parameters(), LR_step2)
    scheduler_step2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_step2, 5, eta_min=0, last_epoch=-1)

    
    if not os.path.exists('./checkpoints_step2'):
        os.mkdir('./checkpoints_step2')
    if not os.path.exists('./pickles_step2'):
        os.mkdir('./pickles_step2')
    if not os.path.exists('./param_unet_'):
        os.mkdir('./param_unet_')


    
    step2_epoch = 0 
 
    
    best_epoch = 0
    best_val_acc = 0
    
    
    ##STEP 1 (Approximation Training)
    for epoch in range(MAX_EPOCH):
        print("Training Step 1: Approximation")
        print(f"Step 1 Epoch #{epoch}")

        train_loss_step1 = 0.0
        net_step1.train()

        with tqdm(enumerate(train_loader_step1), total=len(train_loader_step1)) as tqdm_loader:
            for index, (gt, noisy, red, param) in tqdm_loader:
        
                noisy = noisy.cuda()
                param = param.cuda()
                red = red.cuda()
                
                out = net_step1(noisy, param)
                
                loss_step1 = criterion_step1(out, red)
                train_loss_step1 += float(loss_step1)
                
                
                optimizer_step1.zero_grad()
                loss_step1.backward()
                optimizer_step1.step()
        
        
        scheduler_step1.step()

        psnr_gt_red_list = []
        psnr_gt_net_list = []

        psnr_gt_red_list_train = []
        psnr_gt_net_list_train = []

        train_loss_step1_list.append(train_loss_step1)
        train_epoch_step1_list.append(resume_epoch)
        
        net_step1.eval()
        
         # psnr for train dataset

        for gt, noisy, red, param in tqdm(train_loader_step1):
            noisy_ = noisy.cuda()
            param_ = param.cuda()
            out = net_step1(noisy_, param_)
            

            psnr_gt_red = get_psnr(np.array(gt.cpu().detach()), np.array(red.cpu().detach()))
            psnr_gt_net = get_psnr(np.array(gt.cpu().detach()), np.array(out.cpu().detach()))

            psnr_gt_red_list_train.append(float(psnr_gt_red))
            psnr_gt_net_list_train.append(float(psnr_gt_net))

        # psnr for val dataset
            
        psnr_gt_red_list_test = []
        psnr_gt_net_list_test = []
        p = net_step1.module.return_param_value()
        net_step1.eval()
        param_unet.eval()
        for gt, noisy, red, param in tqdm(val_loader_step1):
            
            param_layer_ = param_unet(noisy)

            res = torch.empty(1, param_layer_.shape[1])

            for idx in range(param_layer_.shape[1]):
                    res[0, idx] = param_layer_[0, idx, :, :].mean()
                        
            out = net_step1(noisy, res)
            
            psnr_gt_red = get_psnr(np.array(gt.cpu().detach()), np.array(red.cpu().detach()))
            psnr_gt_net = get_psnr(np.array(gt.cpu().detach()), np.array(out.cpu().detach()))

            psnr_gt_red_list_test.append(float(psnr_gt_red))
            psnr_gt_net_list_test.append(float(psnr_gt_net))

      
        print('epoch {:03d}, train red vs net : {:.3f} {:.3f}'.format(resume_epoch, float(np.array(psnr_gt_red_list_train).mean()), float(np.array(psnr_gt_net_list_train).mean())))     
        #print('epoch {:03d}, val  red vs net : {:.3f} {:.3f}'.format(resume_epoch, float(np.array(psnr_gt_red_list_test).mean()), float(np.array(psnr_gt_net_list_test).mean())))
        step1_red_psnr_list.append(float(np.array(psnr_gt_red_list_train).mean()))
        step1_net_psnr_list.append(float(np.array(psnr_gt_net_list_train).mean()))
        
        
        avg_val_acc = np.array(psnr_gt_net_list_test).mean()

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            current_save_path = './checkpoints_step1/net_iter{:03d}.pth'.format(resume_epoch+1)
            
            torch.save(net_step1.state_dict(), current_save_path)
            
            best_epoch = resume_epoch
            print("Step 1 model saved")
        
        
        #Train for step 2 every 20 epochs:
        if ((epoch+1) % 20 == 0):
            
        ## Step 2: Optimization 
            print("Training Step 2: Optimization")
       
            print(f"Step 2 Epoch #{step2_epoch}")

            print('./checkpoints_step1/net_iter{:03d}.pth'.format(best_epoch+1))
            
            net_step1_2 = UNETR(in_channels = 3, out_channels = 3, img_size = (512,512), step_flag=2)
            
            saved_state_dict = torch.load('./checkpoints_step1/net_iter{:03d}.pth'.format(best_epoch+1))
            new_state_dict = {key.replace('module.', ''): value for key, value in saved_state_dict.items()}

            net_step1_2.load_state_dict(new_state_dict)
            net_step1_2 = torch.nn.DataParallel(net_step1_2).cuda()
            
            while (step2_epoch <= MAX_EPOCH-1):
                print(f"Step 2 Epoch #{step2_epoch}")

                train_loss_step2 = 0.0
                
                param_unet.train()
                net_step1_2.eval()
                with tqdm(enumerate(train_loader_step1), total=len(train_loader_step1)) as tqdm_loader:
                    for index, (gt, noisy, red, param) in tqdm_loader:
               
                        noisy = noisy.cuda()
                        gt = gt.cuda()
                        param_layer_ = param_unet(noisy)
                        res = torch.empty(1, param_layer_.shape[1])

                        for idx in range(param_layer_.shape[1]):
                            res[0, idx] = param_layer_[0, idx, :, :].mean()
                        
                        
                        out = net_step1_2(noisy, res)
                        loss_step2 = criterion_step2(out, gt)
                        
                        train_loss_step2 += float(loss_step2)

                        
                        
                        optimizer_step2.zero_grad()
                        loss_step2.backward()
                        

                        optimizer_step2.step()
                    
                scheduler_step2.step()

                

                train_loss_step2_list.append(train_loss_step2)
                train_epoch_step2_list.append(step2_epoch)
                
                save_path = './param_unet_/net_iter{:03d}.pth'.format(step2_epoch+1)
                torch.save(param_unet.state_dict(), save_path)

                if((step2_epoch+1)%2 == 0):

                    print("updating dataset")
                    param_unet.eval()
                    with tqdm(enumerate(train_loader_step1), total=len(train_loader_step1)) as tqdm_loader:
                            for index, (gt, noisy, red, param) in tqdm_loader:
                                
                                noisy_temp = noisy[0].permute(1, 2, 0).cpu().numpy() 
                                param_layer_ = param_unet(noisy)

                                res = []
        
                                for idx in range(param_layer_.shape[1]):
                                    res.append(param_layer_[0, idx, :, :].cpu().detach().numpy().mean())
        
                                res = np.array(res) 
                                
                                profile = BM3DProfile()
                                
                                cff = round(normalize(float(res[0].item()), 0, 1, 1, 15), 4)
                                profile.bs_ht = 4 if round(normalize(float(res[1].item()), 0, 1, 4, 8)) < 6 else 8
                                 
                                cspace = 'opp' if (normalize(float(res[2].item()), 0, 1, 0, 1 ) < 0.5) else 'YCbCr'
                                profile.transform_2d_wiener_name  = 'dct' if (normalize(float(res[3].item()), 0, 1, 0, 1 )) < 0.5 else 'dst'
                                profile.bs_wiener = int(normalize(float(res[4].item()), 0, 1, 4, 12))
                                pred_psd = estimate_the_noise(noisy_temp)

                            
                                red_img = bm3d_rgb(noisy_temp, cff * pred_psd[0], profile, colorspace=cspace)
                                
                                red_img = np.minimum(np.maximum(red_img, 0), 1)

                            
                                torch_red = torch.from_numpy(red_img.transpose((2, 0, 1)))
                                
                                dataset.update_data(index, torch_red, float(res[0].item()), float(res[1].item()), float(res[2].item()), float(res[3].item()), float(res[4].item()))
                    
                                    
                    step2_epoch = step2_epoch+1
                    break

                step2_epoch = step2_epoch+1
                
        resume_epoch=resume_epoch+1
    
    
    plt.figure(1)
    plt.plot(train_epoch_step1_list, train_loss_step1_list, label='Step 1 Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Step 1 Training Loss Over Epochs')
    plt.legend()

    
    plt.figure(2)
    plt.plot(train_epoch_step2_list, train_loss_step2_list, label='Step 2 Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Step 2 Training Loss Over Epochs')
    plt.legend()

   
    plt.figure(3)
    plt.plot(train_epoch_step1_list, step1_red_psnr_list, color = 'red', label='BM3D PSNR')
    plt.plot(train_epoch_step1_list, step1_net_psnr_list, color = 'blue', label='Net PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.title('Approximation PSNR')
    plt.legend()

    plt.show()



def test_result():
    image_dir = './SIDD_crop_bm3d'
    loader = test_dataloader(image_dir, batch_size=1, num_threads=16, img_size=512)
    

    param_unet = UNet(in_channels=3, out_channels=5)
    from collections import OrderedDict
    print('./param_unet_/net_iter{:03d}.pth'.format(2))
    state_dict = torch.load('./param_unet_/net_iter{:03d}.pth'.format(2))
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v

    param_unet.load_state_dict(new_state_dict)


    cnt = 0 
    if not os.path.exists('./result'):
        os.mkdir('./result')
    
   
    psnr_gt_red = []
    psnr_gt_before_opt = []

    param_unet.eval()

    for gt, noisy in loader:
        noisy_temp = noisy[0].permute(1, 2, 0).cpu().numpy() 
        
        param_layer = param_unet(noisy)

      

        profile = BM3DProfile()
       
        cff = round(normalize(float(param_layer[0, 0, :, :].mean().cpu().detach().numpy()), 0, 1, 1, 15), 4)
        profile.bs_ht = 4 if round(normalize(float(param_layer[0, 1, :, :].mean().cpu().detach().numpy()), 0, 1, 4, 8)) < 6 else 8
        cspace = 'opp' if (normalize(float(param_layer[0, 2, :, :].mean().cpu().detach().numpy()), 0, 1, 0, 1 ) < 0.5) else 'YCbCr'
        profile.transform_2d_wiener_name  = 'dct' if (normalize(float(param_layer[0, 3, :, :].mean().cpu().detach().numpy()), 0, 1, 0, 1 )) < 0.5 else 'dst'
        profile.bs_wiener = round(normalize(float(param_layer[0, 4, :, :].mean().cpu().detach().numpy()), 0, 1, 4, 15))
        pred_psd = estimate_the_noise(noisy_temp)
     
        red_img = bm3d_rgb(noisy_temp, cff * pred_psd[0], profile, colorspace=cspace)
        
        red_out = np.minimum(np.maximum(red_img, 0), 1)

        
        torch_red = torch.from_numpy(red_out.transpose((2, 0, 1)))
     
        print(float(get_psnr(np.array(gt.cpu().detach()), np.array(torch_red.cpu().detach()))))
        
        psnr_gt_red.append(float(get_psnr(np.array(gt.cpu().detach()), np.array(torch_red.cpu().detach()))))
        psnr_gt_before_opt.append(float(get_psnr(np.array(gt.cpu().detach()), np.array(noisy.cpu().detach()))))
        
        
        torch_red = torch_red.unsqueeze(0)
        gt = np.array(gt)[0].transpose(1, 2, 0)
        noisy = np.array(noisy)[0].transpose(1, 2, 0)
        red_opt = np.array(torch_red)[0].transpose(1, 2, 0)
        
        fig = plt.figure()
        plt.subplot(221)
        plt.imshow(gt)
        plt.subplot(222)
        plt.imshow(noisy)
    
        plt.subplot(223)
        plt.imshow(red_opt)

        if not os.path.exists('./result/figure'):
            os.mkdir('./result/figure')
        fig.savefig('./result/figure/{}.png'.format(cnt))
        cnt += 1

    with open('./result/psnr_.txt', 'w') as f:
        for idx in range(len(psnr_gt_red)):
            f.write('psnr gt-red vs gt-noisy: {:.3f} {:.3f}\n'.format(psnr_gt_red[idx], psnr_gt_before_opt[idx]))
        f.write('\navg  gt-red vs gt-noisy: {:.3f} {:.3f}\n'.format(np.array(psnr_gt_red).mean(), np.array(psnr_gt_before_opt).mean()))
    f.close()



if __name__ == '__main__':
    train()
    #test_result()
    
