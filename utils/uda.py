import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2

import numpy as np

from config import *

def dacs(xs,ys,xt,yt,ratio =0.25):
    b,c = yt.shape[:2]

    xm = xt.detach().clone() #b,c,h,w
    ym = yt.detach().clone() #b,c,h,w

    for i in range(b):
        cls = ys[i].unique()

        ys_one_hot = F.one_hot(ys,c).permute(0,3,1,2) # b,c,h,w
        random_indices = torch.randperm(len(cls))[:int(len(cls)*ratio)]

        seleceted_cls = cls[random_indices]

        region = torch.zeros_like(ys)[0]
        for j in seleceted_cls:
            region = (region | (ys[i]==j))

        xm[i] = xm[i]*(1-region) + xs[i]*region
        ym[i] = ym[i]*(1-region) + ys_one_hot[i]*region

    return xm, ym


def masking(input,mask_ratio = 0.5 , mask_size = 32):
    b,c,h,w = input.shape

    h_patch = h // mask_size
    w_patch = w // mask_size

    mask = (np.random.uniform(0,1,(h_patch,w_patch,b)) > mask_ratio).astype(np.uint8())

    mask = cv2.resize(mask,(w,h),interpolation = cv2.INTER_NEAREST)
    mask = torch.from_numpy(mask).permute(2,0,1).reshape(b,1,h,w).to(input.device)

    output = input.detach().clone() * mask

    return output

class target_loss(nn.Module):
    def __init__(self,t=0.968):
        super(target_loss,self).__init__()
        self.t = t
        self.celoss = nn.CrossEntropyLoss(reduction='none')

    def forward(self,pred, pseudo_label):

        b,c,h,w = pseudo_label.shape

        confidence, pt = torch.max(pseudo_label, dim=1) # b,h,w

        qt = (torch.sum(confidence.view(b,-1) > self.t,1) / (h*w)).view(b,1,1) # b,1,1

        loss = torch.mean(self.celoss(pred, pt) * qt) # b,h,w * b,1,1 -> b,h,w -> 1

        return loss
    
def slide_pred(model,img,window_size=(CROP_HEIGHT,CROP_WIDTH),stride=(CROP_HEIGHT,CROP_WIDTH),softmax=False,padding=False):
    #추론 속도를 높이려면 입력 이미지를 나누어 Batch로 만들도록 코드를 재구성
    #학습, 추론 과정 모두 Batch 단위로 입력이 들어오기 때문에 메모리 문제로 인해 Batch로 만들지 못함
    #실제 적용에서는 무조건 Batch가 1이기 때문에 이미지 하나를 Batch로 만들 수 있음
    b,c,h,w = img.shape


    if padding :
        padded_img = torch.zeros((b,c,h+(window_size[0]-stride[0])*2,w+(window_size[1]-stride[1])*2)).to(input.device)
        padded_img[:,:,window_size[0]-stride[0]:-(window_size[0]-stride[0]),window_size[1]-stride[1]:-(window_size[1]-stride[1])] = img
        output = torch.zeros((b,14,h+(window_size[0]-stride[0])*2,w+(window_size[1]-stride[1])*2)).to(input.device)
    else:
        padded_img = img
        output = torch.zeros((b,14,h,w)).to(input.device)

    ph,pw = padded_img.shape[2:]

    for i in range(0,ph-window_size[0]+1,stride[0]):
        for j in range(0,pw-window_size[1]+1,stride[1]):
            input = padded_img[:,:,i:i+window_size[0],j:j+window_size[1]]
            if softmax :
                pred = torch.softmax(model(input)['out'],1)
            else:
                pred = model(input)['out']

            output[:,:13,i:i+window_size[0],j:j+window_size[1]] += pred
            output[:,13,i:i+window_size[0],j:j+window_size[1]] += 1

    output = output[:,:13] / output[:,13:]

    if padding:
        output = output[:,:,window_size[0]-stride[0]:-(window_size[0]-stride[0]),window_size[1]-stride[1]:-(window_size[1]-stride[1])]

    return output