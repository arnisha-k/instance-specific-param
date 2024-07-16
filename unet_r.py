# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");

from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.nets import ViT
import numpy as np

def normalize(x, a, b, c, d):
    '''
    x from (a, b) to (c, d)
    '''
    return (float(x) - a) * (float(d) - c) / (float(b) - a) + float(c)


class UNETR(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        step_flag : int,
        img_size: Tuple[int, int],
        feature_size: int = 32,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = False,
        res_block: bool = True,
        dropout_rate: float = 0.0,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.

        Examples::

            # for single channel input 4-channel output with patch size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')

            # for 4-channel input 3-channel output with patch size of (128,128,128), conv position embedding and instance norm
            >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), pos_embed='conv', norm_name='instance')

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")
        self.step_flag = step_flag
        self.feature_size= feature_size
        self.img_size = img_size
        self.size = img_size[0]
        self.num_layers = 12
        self.patch_size = (16, 16)
        self.feat_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
            #img_size[2] // self.patch_size[2],
        )
        self.param_layer = torch.cat([
            torch.rand([1]).expand(1, 1, self.size, self.size),
            torch.tensor(normalize(torch.randint(1, 3, [1]) * 4., 4, 8, 0, 1)).expand(1, 1, self.size, self.size),
            torch.tensor(normalize(torch.randint(0, 2, [1]), 0, 1, 0, 1)).expand(1, 1, self.size, self.size),
            torch.tensor(normalize(torch.randint(0, 2, [1]), 0, 1, 0, 1)).expand(1, 1, self.size, self.size),
            torch.tensor(normalize(torch.randint(4, 16, [1]), 4, 15, 0, 1)).expand(1, 1, self.size, self.size)
        ], dim=1).cuda()
        #self.param_layer = torch.cat([
        #    torch.tensor(normalize(11., 1, 15, 0, 1)).expand(1, 1, self.size, self.size),
        #    torch.tensor(normalize(8., 4, 8, 0, 1)).expand(1, 1, self.size, self.size),
        #    torch.tensor(normalize(1, 0, 1, 0, 1)).expand(1, 1, self.size, self.size),
        #    torch.tensor(normalize(0, 0, 1, 0, 1)).expand(1, 1, self.size, self.size),
        #    torch.tensor(normalize(6., 3, 15, 0, 1)).expand(1, 1, self.size, self.size)
        #], dim=1).cuda()

        if self.step_flag == 1:
            for p in self.parameters():
                p.requires_grad = True
        elif self.step_flag > 1:
            for p in self.parameters():
                p.requires_grad = False
        if self.step_flag == 3:
                self.param_layer.requires_grad = True
        elif self.step_flag <= 2 or self.step_flag > 3:
                self.param_layer.requires_grad = False  # For approximation, param layer_grad = False

        self.hidden_size = hidden_size
        self.classification = False
        self.vit = ViT(
            in_channels=in_channels+5,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate, spatial_dims=2
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=2,
            in_channels=in_channels+5,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=2,
            in_channels=hidden_size+5,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=2,
            in_channels=hidden_size+5,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=2,
            in_channels=hidden_size+5,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder5 = UnetrPrUpBlock(
            spatial_dims=2,
            in_channels=hidden_size,
            out_channels=feature_size * 16,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder6 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=hidden_size,
            out_channels=feature_size * 16,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.avgpool = nn.AvgPool2d(kernel_size=16, stride=16)
        
        self.out = UnetOutBlock(spatial_dims=2, in_channels=feature_size, out_channels=out_channels) 
    
    def return_param_layer(self):
        return self.param_layer

    def load_param_layer(self, value):
        self.param_layer = value
    
    #returns the parameter layer by taking average across the parameter channels
    def return_param_value(self):
        res = []
        for idx in range(self.param_layer.shape[1]):
            res.append(self.param_layer[0, idx, :, :].cpu().detach().numpy().mean())
        return np.array(res)
    
    def return_param(self):
        res = []
        for idx in range(self.param_layer.shape[1]):
            res.append(self.param_layer[0, idx, :, :].cpu().detach().numpy().mean())
        return np.array(res)

    def update_param(self):
        self.param_layer.requires_grad=False
        self.param_layer = torch.clamp(self.param_layer.clone(), 0, 1)
        self.param_layer.requires_grad=True
        
    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], hidden_size)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def load_from(self, weights):
        with torch.no_grad():
            res_weight = weights
            # copy weights from patch embedding
            for i in weights["state_dict"]:
                print(i)
            self.vit.patch_embedding.position_embeddings.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.position_embeddings_3d"]
            )
            self.vit.patch_embedding.cls_token.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.cls_token"]
            )
            self.vit.patch_embedding.patch_embeddings[1].weight.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.patch_embeddings.1.weight"]
            )
            self.vit.patch_embedding.patch_embeddings[1].bias.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.patch_embeddings.1.bias"]
            )

            # copy weights from  encoding blocks (default: num of blocks: 12)
            for bname, block in self.vit.blocks.named_children():
                print(block)
                block.loadFrom(weights, n_block=bname)
            # last norm layer of transformer
            self.vit.norm.weight.copy_(weights["state_dict"]["module.transformer.norm.weight"])
            self.vit.norm.bias.copy_(weights["state_dict"]["module.transformer.norm.bias"])

    def forward(self, x_in, param_list=None):
        param_list_hidden = param_list
        if self.step_flag <= 2:
               param_list = param_list.view(param_list.size(0), param_list.size(1), 1, 1)   
               param_list = param_list.repeat(1, 1, x_in.size(2), x_in.size(3))
               param_layer = param_list
               self.param_layer = param_layer
               
        if self.step_flag <= 2 :
               x, hidden_states_out = self.vit(torch.cat([x_in, param_layer], dim=1))
        elif self.step_flag > 2: 
               x, hidden_states_out = self.vit(torch.cat([x_in,self.param_layer.repeat(x_in.size(0), 1, 1, 1)], dim=1))
        
        # Parameter layer concatenated with input and hidden states 3, 6 and 9
        x1 = x_in
        if self.step_flag <= 2:
               x1 = torch.cat([x1, param_layer], dim=1)
        elif self.step_flag > 2:
               x1 = torch.cat([x1, self.param_layer.repeat(x_in.size(0), 1, 1, 1)], dim=1)
        enc1 = self.encoder1(x1)
        x2 = hidden_states_out[3]
        x2 = self.proj_feat(x2, self.hidden_size, self.feat_size)
        if self.step_flag <= 2:
               param_hidden_layer = param_list_hidden.view(param_list_hidden.size(0), param_list_hidden.size(1), 1, 1)
               param_hidden_layer = param_hidden_layer.repeat(1, 1, x2.size(2), x2.size(3))
               param_layer = param_hidden_layer
               
               x2 = torch.cat([x2, param_layer], dim=1)
        elif self.step_flag > 2:
               x2 = torch.cat([x2, self.avgpool(self.param_layer).repeat(x2.shape[0], 1, 1, 1)], dim=1)
        
        enc2 = self.encoder2(x2)
        x3 = hidden_states_out[6]
        x3 = self.proj_feat(x3, self.hidden_size, self.feat_size)
        if self.step_flag <= 2:
               param_hidden_layer = param_list_hidden.view(param_list_hidden.size(0), param_list_hidden.size(1), 1, 1)
               param_hidden_layer = param_hidden_layer.repeat(1, 1, x3.size(2), x3.size(3))
               param_layer = param_hidden_layer
               x3 = torch.cat([x3, param_layer], dim=1)
        elif self.step_flag > 2:
               x3 = torch.cat([x3, self.avgpool(self.param_layer).repeat(x3.shape[0], 1, 1, 1)], dim=1)
        
        enc3 = self.encoder3(x3)
        x4 = hidden_states_out[9]
        x4 = self.proj_feat(x4, self.hidden_size, self.feat_size)
        if self.step_flag <= 2:
               param_hidden_layer = param_list_hidden.view(param_list_hidden.size(0), param_list_hidden.size(1), 1, 1)
               param_hidden_layer = param_hidden_layer.repeat(1, 1, x4.size(2), x4.size(3))
               param_layer = param_hidden_layer
               x4 = torch.cat([x4, param_layer], dim=1)
        elif self.step_flag > 2:
               x4 = torch.cat([x4, self.avgpool(self.param_layer).repeat(x4.shape[0], 1, 1, 1)], dim=1)
       
        
        enc4 = self.encoder4(x4)
        dec4 = self.proj_feat(x, self.hidden_size, self.feat_size)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        logits = self.out(out)
        
        return logits
