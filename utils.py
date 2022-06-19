import cv2
from torchvision.transforms import functional as Ft
import numpy as np
import torch
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import torch.optim as optim

import wandb

wandb.init(project='debugging-nn',name=f"RRDB_MSE_LR0.0001_NEP255_LOSSWT1._WAND")

torch.manual_seed(123) 

#utils
def convert_image(in_tensor):
    in_tensor = in_tensor.squeeze(0)
    gray_scale = torch.sum(in_tensor,0)
    gray_scale = gray_scale / in_tensor.shape[0]
    gray_scale = gray_scale.detach().to('cpu').numpy()
    gray_scale = gray_scale*255.
    gray_scale =gray_scale.astype(np.uint8)
    return gray_scale

def convert_image_wo_sum(in_tensor):
    in_tensor = in_tensor.squeeze(0)
    gray_scale = in_tensor[0].squeeze(0)
    gray_scale = gray_scale.detach().to('cpu').numpy()
    gray_scale = gray_scale*255.
    gray_scale =gray_scale.astype(np.uint8)
    return gray_scale

'''reduce learning rate of optimizer by half on every  150 and 225 epochs'''
def adjust_learning_rate(optimizer, epoch,lr):
    if epoch % 150 == 0 or epoch % 250==0:
        lr = lr * 0.5
    # log to TensorBoard
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def load_image_to_tensor():
    
    lr_2_i = cv2.imread("images/lr_f1_160_2_z_47.png").astype(np.float32) / 255.
    lr_4_i = cv2.imread("images/lr_f1_160_4_z_47.png").astype(np.float32) / 255.
    lr_6_i = cv2.imread("images/lr_f1_160_6_z_47.png").astype(np.float32) / 255.
    lr_8_i = cv2.imread("images/lr_f1_160_8_z_47.png").astype(np.float32) / 255.
    hr_i = cv2.imread("images/hr_f1_160_z_47.png").astype(np.float32) / 255.

    lr_2_i = cv2.cvtColor(lr_2_i, cv2.COLOR_BGR2GRAY)
    lr_4_i = cv2.cvtColor(lr_4_i,cv2.COLOR_BGR2GRAY)
    lr_6_i = cv2.cvtColor(lr_6_i,cv2.COLOR_BGR2GRAY)
    lr_8_i = cv2.cvtColor(lr_8_i,cv2.COLOR_BGR2GRAY)
    hr_i = cv2.cvtColor(hr_i,cv2.COLOR_BGR2GRAY)

    lr_2 = Ft.to_tensor(lr_2_i).unsqueeze(0)
    lr_4 = Ft.to_tensor(lr_4_i).unsqueeze(0)
    lr_6 = Ft.to_tensor(lr_6_i).unsqueeze(0)
    lr_8 = Ft.to_tensor(lr_8_i).unsqueeze(0)
    hr = Ft.to_tensor(hr_i).unsqueeze(0)
    return {'lr_2':lr_2,'lr_4':lr_4,'lr_6':lr_6,'lr_8':lr_8,'hr':hr}