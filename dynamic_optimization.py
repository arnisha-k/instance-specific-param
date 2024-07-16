from unet_r import UNETR
import numpy as np
import torch
from tools.dataloader import train_dataloader, val_dataloader, test_dataloader, MyDataset
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
from skimage.metrics import structural_similarity as ssim

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


def normalize(x, a, b, c, d):
    '''
    x from (a, b) to (c, d)
    '''
    return (float(x) - a) * (float(d) - c) / (float(b) - a) + float(c)



def train(resume_epoch=None):
    
    MAX_EPOCH = 45
    LR = 0.005
    batch_size = 1
    
    print("Loading Data")
    
    image_dir = './SIDD_crop_bm3d'
    
    transform = transforms.Compose([ transforms.ToTensor(),])
    dataset = MyDataset(img_path=image_dir, data_transforms=transform, is_train=True, img_size=512)
    
    train_loader_step1 = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                                                num_workers=16, pin_memory=True)
    
    
    val_loader_step1 = val_dataloader(image_dir, batch_size=1, num_threads=16, img_size=512)
    
    LR_step2 = 0.02
    

    #Approximation Model (Step 1) Initialization
    net_step1 = UNETR(in_channels = 3, out_channels = 3, img_size = (512,512), step_flag=1)
    net_step1 = torch.nn.DataParallel(net_step1).cuda()

    criterion_step1 = torch.nn.L1Loss()
    
    optimizer_step1 = torch.optim.Adam(net_step1.parameters(), LR)
    
    scheduler_step1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_step1, 20, eta_min=0, last_epoch=-1)
    profile = BM3DProfile()
    
        
    if not os.path.exists('./checkpoints_step1'):
        os.mkdir('./checkpoints_step1')

    if not os.path.exists('./pickles_step1'):
        os.mkdir('./pickles_step1')
    
    resume_epoch = 0
    #Optimization Initialization (Step 2) Steps
    
    net_step2 = UNETR(in_channels = 3, out_channels = 3, step_flag = 3, img_size = (512,512))
    net_step2 = torch.nn.DataParallel(net_step2).cuda()
   
    criterion_step2 = torch.nn.MSELoss()
    
    if not os.path.exists('./checkpoints_step2'):
        os.mkdir('./checkpoints_step2')
    
    if not os.path.exists('./pickles_step2'):
        os.mkdir('./pickles_step2')


    step2_epoch = 0 
   
    #Training 
    best_val_acc_step2 = 0


    #Step 1 (Approximation Training)
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

        for gt, noisy, red, param in tqdm(train_loader_step1):
            
            noisy_ = noisy.cuda()
            param_ = param.cuda()
            out = net_step1(noisy_, param_)
            

            psnr_gt_red = get_psnr(np.array(gt.cpu().detach()), np.array(red.cpu().detach()))
            psnr_gt_net = get_psnr(np.array(gt.cpu().detach()), np.array(out.cpu().detach()))

            psnr_gt_red_list_train.append(float(psnr_gt_red))
            psnr_gt_net_list_train.append(float(psnr_gt_net))

      
            
        psnr_gt_red_list_test = []
        psnr_gt_net_list_test = []
        p = net_step1.module.return_param_value()     
        p =  torch.tensor(p)
        p = p.view(1, -1)
        net_step1.eval()
        for gt, noisy, red, param in tqdm(val_loader_step1):
            noisy_ = noisy.cuda()
            param_ = param.cuda()
            out = net_step1(noisy_, p)
            psnr_gt_red = get_psnr(np.array(gt.cpu().detach()), np.array(red.cpu().detach()))
            psnr_gt_net = get_psnr(np.array(gt.cpu().detach()), np.array(out.cpu().detach()))

            psnr_gt_red_list_test.append(float(psnr_gt_red))
            psnr_gt_net_list_test.append(float(psnr_gt_net))

      
        print('epoch {:03d}, train red vs net : {:.3f} {:.3f}'.format(resume_epoch, float(np.array(psnr_gt_red_list_train).mean()), float(np.array(psnr_gt_net_list_train).mean())))     
        #print('epoch {:03d}, val  red vs net : {:.3f} {:.3f}'.format(resume_epoch, float(np.array(psnr_gt_red_list_test).mean()), float(np.array(psnr_gt_net_list_test).mean())))
        
        step1_red_psnr_list.append(float(np.array(psnr_gt_red_list_train).mean()))
        step1_net_psnr_list.append(float(np.array(psnr_gt_net_list_train).mean()))
        
        current_save_path = './checkpoints_step1/net_iter{:03d}.pth'.format(resume_epoch+1)
        
        torch.save(net_step1.state_dict(), current_save_path)
        

      
        if ((epoch+1) % 20 == 0):
            
        ## Step 2: Optimization 
            print("Training Step 2: Optimization")   
            
            while (step2_epoch <= MAX_EPOCH-1):
                print(f"Step 2 Epoch #{step2_epoch}")

                print('./checkpoints_step1/net_iter{:03d}.pth'.format(resume_epoch+1))

                net_step2.load_state_dict(torch.load('./checkpoints_step1/net_iter{:03d}.pth'.format(resume_epoch+1)))

                train_loss_step2 = 0.0
                net_step2.eval()
                
                
                with tqdm(enumerate(train_loader_step1), total=len(train_loader_step1)) as tqdm_loader:
                    for index, (gt, noisy, red, param) in tqdm_loader:           
                        optimizer_step2 = torch.optim.Adam([net_step2.module.param_layer], LR_step2)
                        noisy = noisy.cuda()
                        gt = gt.cuda()
                        out = net_step2(noisy)
                        # compute loss.
                        loss_step2 = criterion_step2(out, gt)          
                        train_loss_step2 += float(loss_step2)
                        optimizer_step2.zero_grad()
                        loss_step2.backward()
                        for idx in range(5):
                            avg_grad = net_step2.module.param_layer.grad.data[0, idx, :, :].mean().float()
                            net_step2.module.param_layer.grad.data[0, idx, :, :] = avg_grad

                        optimizer_step2.step()
                        net_step2.module.update_param()
                        # Update Dataset
                        noisy_temp = noisy[0].permute(1, 2, 0).cpu().numpy() 
                        res = net_step2.module.return_param_value()
                        profile = BM3DProfile()
                        cff = round(normalize(float(res[0].item()), 0, 1, 1, 20), 4)
                        profile.bs_ht = 4 if round(normalize(float(res[1].item()), 0, 1, 4, 8)) < 6 else 8 
                        cspace = 'opp' if (normalize(float(res[2].item()), 0, 1, 0, 1 ) < 0.5) else 'YCbCr'
                        profile.transform_2d_wiener_name  = 'dct' if (normalize(float(res[3].item()), 0, 1, 0, 1 )) < 0.5 else 'dst'
                        profile.bs_wiener = int(normalize((res[4].item()), 0, 1, 3, 15))
                        pred_psd = estimate_the_noise(noisy_temp)
                        red_img = bm3d_rgb(noisy_temp, cff * pred_psd[0], profile, colorspace=cspace)
                        red_img = np.minimum(np.maximum(red_img, 0), 1)
                        torch_red = torch.from_numpy(red_img.transpose((2, 0, 1)))
                        dataset.update_data(index, torch_red, float(res[0].item()), float(res[1].item()), float(res[2].item()), float(res[3].item()), float(res[4].item()))
                    
                psnr_gt_red_list = []
                psnr_gt_net_list = []

                psnr_gt_red_list_test = []
                psnr_gt_net_list_test = []

                train_loss_step2_list.append(train_loss_step2)
                train_epoch_step2_list.append(step2_epoch)

                net_step2.eval()

                for gt, noisy, red, param in tqdm(train_loader_step1):
                    noisy_ = noisy.cuda()
                    out = net_step2(noisy_)

                    psnr_gt_red = get_psnr(np.array(gt.cpu().detach()), np.array(red.cpu().detach()))
                    psnr_gt_net = get_psnr(np.array(gt.cpu().detach()), np.array(out.cpu().detach()))

                    psnr_gt_red_list.append(float(psnr_gt_red))
                    psnr_gt_net_list.append(float(psnr_gt_net))

                res = net_step2.module.return_param_value()
                
                net_step2.eval()
                for gt, noisy, red, param in tqdm(val_loader_step1):
                    noisy_ = noisy.cuda()
                    out = net_step2(noisy_)

                    psnr_gt_red = get_psnr(np.array(gt.cpu().detach()), np.array(red.cpu().detach()))
                    psnr_gt_net = get_psnr(np.array(gt.cpu().detach()), np.array(out.cpu().detach()))

                    psnr_gt_red_list_test.append(float(psnr_gt_red))
                    psnr_gt_net_list_test.append(float(psnr_gt_net))
                
                #if (epoch+1) % 2 == 0:
                #    LR_step2 *= 0.8
                #if (epoch+1) % 30 == 0:
                #    LR_step2 = 0.002

                print('epoch {:03d}, red vs net (train) : {:.3f} {:.3f}'.format(step2_epoch, float(np.array(psnr_gt_red_list).mean()), float(np.array(psnr_gt_net_list).mean())))
                #print('epoch {:03d}, red vs net (val) : {:.3f} {:.3f}'.format(step2_epoch, float(np.array(psnr_gt_red_list_test).mean()), float(np.array(psnr_gt_net_list_test).mean())))
                step2_red_psnr_list.append(float(np.array(psnr_gt_red_list).mean()))
                step2_net_psnr_list.append(float(np.array(psnr_gt_net_list).mean()))
                step2_psnr_gt_red_list_test_mean.append(float(np.array(psnr_gt_red_list_test).mean()))
                step2_psnr_gt_net_list_test_mean.append(float(np.array(psnr_gt_net_list_test).mean()))
                

                avg_val_acc_step2 = np.array(psnr_gt_net_list_test).mean()
                
                net_step1.module.param_layer.copy_(net_step2.module.return_param_layer())           
                
                if avg_val_acc_step2 > best_val_acc_step2:
                    best_val_acc = avg_val_acc_step2
                    save_path = './checkpoints_step2/net_iter{:03d}.pth'.format(step2_epoch+1)
                    torch.save(net_step2.state_dict(), save_path)
                    param_layer_value = net_step2.module.return_param_layer()
                    f = open('./pickles_step2/param_layer{:03d}.pkl'.format(step2_epoch+1), 'wb')
                    pickle.dump(param_layer_value, f)
                    f.close()
                    print("Step 2 model saved")

                
                step2_epoch = step2_epoch+1
                break
                  
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
    
    plt.figure(4)
    plt.plot(train_epoch_step2_list, step2_red_psnr_list, color = 'red', label='BM3D PSNR')
    plt.plot(train_epoch_step2_list, step2_net_psnr_list, color = 'blue', label='Net PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.title('Optimization PSNR')
    plt.legend()

    plt.show()

def test_result():
    image_dir = './SIDD_crop_bm3d'
    loader = test_dataloader(image_dir, batch_size=1, num_threads=16, img_size=512)
    
    f = open('./pickles_step2/param_layer{:03d}.pkl'.format(4), 'rb')
    param_layer = pickle.load(f)
    print(param_layer[0, 0, :, :].mean().cpu().detach().numpy(), end=' ')
    print(param_layer[0, 1, :, :].mean().cpu().detach().numpy(), end=' ')
    print(param_layer[0, 2, :, :].mean().cpu().detach().numpy(), end=' ')
    print(param_layer[0, 3, :, :].mean().cpu().detach().numpy(), end=' ')
    print(param_layer[0, 4, :, :].mean().cpu().detach().numpy())
    
    f.close()
    
    cnt = 0 
    if not os.path.exists('./result'):
        os.mkdir('./result')
    
    
    psnr_gt_red = []
    psnr_gt_before_opt = []
    ssim_gt_red = []
    ssim_gt_before_opt = []
    
    for gt, noisy in loader:
        noisy_temp = noisy[0].permute(1, 2, 0).cpu().numpy() 
        profile = BM3DProfile()
        cff = round(normalize(float(param_layer[0, 0, :, :].mean().cpu().detach().numpy()), 0, 1, 1, 15), 4)
        profile.bs_ht = 4 if round(normalize(float(param_layer[0, 1, :, :].mean().cpu().detach().numpy()), 0, 1, 4, 8)) < 6 else 8
            #profile.bs_ht = round(normalize(float(res[1].item()), 0, 1, 4, 8)) 
        cspace = 'opp' if (normalize(float(param_layer[0, 2, :, :].mean().cpu().detach().numpy()), 0, 1, 0, 1 ) < 0.5) else 'YCbCr'
        profile.transform_2d_wiener_name  = 'dct' if (normalize(float(param_layer[0, 3, :, :].mean().cpu().detach().numpy()), 0, 1, 0, 1 )) < 0.5 else 'dst'
        profile.bs_wiener = round(normalize(float(param_layer[0, 4, :, :].mean().cpu().detach().numpy()), 0, 1, 4, 15))
        pred_psd = estimate_the_noise(noisy_temp)
        
        red_img = bm3d_rgb(noisy_temp, cff * pred_psd[0], profile, colorspace=cspace)
        
        red_out = np.minimum(np.maximum(red_img, 0), 1)

        
        torch_red = torch.from_numpy(red_out.transpose((2, 0, 1)))
        
        
        psnr_gt_red.append(float(get_psnr(np.array(gt.cpu().detach()), np.array(torch_red.cpu().detach()))))
        psnr_gt_before_opt.append(float(get_psnr(np.array(gt.cpu().detach()), np.array(noisy.cpu().detach()))))
        ssim_gt_red.append(ssim((np.array(gt)[0].transpose(1, 2, 0)), (np.array(torch_red).transpose(1, 2, 0)), data_range=1.0, channel_axis=-1))
        ssim_gt_before_opt.append(ssim((np.array(gt)[0].transpose(1, 2, 0)), (np.array(noisy)[0].transpose(1, 2, 0)), data_range=1.0, channel_axis=-1))
        
        
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

    with open('./result/psnr_test.txt', 'w') as f:
        for idx in range(len(psnr_gt_red)):
            f.write('psnr gt-red vs gt-noisy: {:.3f} {:.3f}\n'.format(psnr_gt_red[idx], psnr_gt_before_opt[idx]))
        f.write('\navg  gt-red vs gt-noisy: {:.3f} {:.3f}\n'.format(np.array(psnr_gt_red).mean(), np.array(psnr_gt_before_opt).mean()))
    f.close()

    with open('./result/ssim_test.txt', 'w') as f:
        for idx in range(len(ssim_gt_red)):
            f.write('SSIM gt-red vs gt-noisy: {:.3f} {:.3f}\n'.format(ssim_gt_red[idx], ssim_gt_before_opt[idx]))
        f.write('\navg  gt-red vs gt-noisy: {:.3f} {:.3f}\n'.format(np.array(ssim_gt_red).mean(), np.array(ssim_gt_before_opt).mean()))
    f.close()


if __name__ == '__main__':
    train()
    #test_result()
    
