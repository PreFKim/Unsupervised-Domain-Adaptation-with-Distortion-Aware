import argparse
import tqdm
import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .utils.uda import slide_pred
from .dataset.dataset import CustomDataset
from .config import *
from .model.daformer import (
    mit_b0, 
    mit_b1, 
    mit_b2, 
    mit_b3, 
    mit_b4, 
    mit_b5, 
    daformerhead,
    daformer
    )
from .model.unet import UNet, ResUNet
from .model.deeplabv2 import deeplabv2
from .Deeplabv3 import network

def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

#Deeplab v2
#model = deeplabv2(torchvision.models.segmentation.deeplabv3_resnet101(num_classes=13,weights_backbone= 'ResNet101_Weights.IMAGENET1K_V1').backbone )
#teacher_model = deeplabv2(torchvision.models.segmentation.deeplabv3_resnet101(num_classes=13,weights_backbone= 'ResNet101_Weights.IMAGENET1K_V1').backbone)

#Deeplab v3+ ImageNet Pretrain
#model = torchvision.models.segmentation.deeplabv3_resnet101(num_classes=13, weights_backbone= 'ResNet101_Weights.IMAGENET1K_V1')
#teacher_model = torchvision.models.segmentation.deeplabv3_resnet101(num_classes=13, weights_backbone= 'ResNet101_Weights.IMAGENET1K_V1')

#Deeplab v3+ CityScape Pretrain (https://github.com/VainF/DeepLabV3Plus-Pytorch)
#model = network.modeling.__dict__['deeplabv3plus_resnet101'](num_classes=13, output_stride=8)
#teacher_model = network.modeling.__dict__['deeplabv3plus_resnet101'](num_classes=13, output_stride=8)
#state_dict= torch.load(root_path + '/pretrain/best_deeplabv3plus_resnet101_cityscapes_os16.pth')['model_state']
#filtered_dict = {key: value for key, value in state_dict.items() if 'backbone' in key} #encoder만 가져오기
#model.load_state_dict(filtered_dict,strict=False )

#VGG UNET
#model = UNet(torchvision.models.vgg13_bn(weights= 'IMAGENET1K_V1'),512)
#teacher_model = UNet(torchvision.models.vgg13_bn(weights= 'IMAGENET1K_V1'),512)

## Res UNET
#model = ResUNet(torchvision.models.resnet101(weights= 'ResNet101_Weights.IMAGENET1K_V1'),2048)
#teacher_model = ResUNet(torchvision.models.resnet101(weights= 'ResNet101_Weights.IMAGENET1K_V1'),2048)

## DaFormer (Weight from Segformer : https://github.com/NVlabs/SegFormer)
# b3,b4,b5의 선택지가 있음
model = daformer(mit_b4(),daformerhead())
model.load_state_dict(torch.load(f'./pretrain/segformer.b4.1024x1024.city.160k.pth')['state_dict'],strict=False)
teacher_model = daformer(mit_b4(),daformerhead())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type="str", default="./data(resized)/")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--checkpoint", type=str, default="")
    args = parser.parse_args()

    device = 'cuda'

    state_dict = torch.load(args.checkpoint, device)
    model.load_state_dict(state_dict['model'])
    print(state_dict['epoch'])
    test_set = CustomDataset(csv_file=f'./test.csv',
                            data_path=args.data_path,
                            infer=True
                            )
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


    print(state_dict['epoch'])
    model.to(device)
    model.eval()
    with torch.no_grad():
        result = []
        for i,imgs in enumerate(tqdm.tqdm(test_loader)):
            b,c,h,w = imgs.shape

            input = imgs.float().to(device)

            #preds = model(input)['out'] #전체 이미지를 활용한 모델은 이 코드 사용
            preds = slide_pred(model,input,stride=(CROP_HEIGHT//2,CROP_WIDTH//2),softmax = True ,padding=False) # b,c,h,w


            original_mask = torch.from_numpy(original_mask).reshape(1,h,w).repeat(b,1,1)
            pred_original = torch.argmax(preds,1).cpu().numpy() # b,c,h,w -> b,h,w



            pred_resize = F.interpolate(preds, size=(540,960),mode='bilinear',align_corners=False)
            pred_resize = torch.argmax(pred_resize,1).cpu().numpy() # b,c,h,w -> b,h,w


            #타원을 제외한 위치에 있는 값들 후처리
            center = (960//2, int(540*0.375))  # x,y
            axis_length = (960//2, int(540*0.64))  # 장축 반지름과 단축 반지름
            resize_mask = np.ones((540,960),dtype=np.uint8)
            cv2.ellipse(resize_mask, center, axis_length, 0, 0, 360, 0, -1)
            resize_mask = torch.from_numpy(resize_mask).unsqueeze(0).repeat(b,1,1)

            pred_resize[resize_mask==1] = 12

            for j,pred in enumerate(pred_resize):
                for class_id in range(12):
                    class_mask = (pred == class_id).astype(np.uint8)
                    if np.sum(class_mask) > 0: # 마스크가 존재하는 경우 encode
                        mask_rle = rle_encode(class_mask)
                        result.append(mask_rle)
                    else: # 마스크가 존재하지 않는 경우 -1
                        result.append(-1)

    print(state_dict['epoch'])
    submit = pd.read_csv(f'./sample_submission.csv')
    submit['mask_rle'] = result
    submit.to_csv(f'./baseline_submit.csv', index=False)
    submit.head()