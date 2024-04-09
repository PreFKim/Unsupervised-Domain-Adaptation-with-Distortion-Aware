
import os
import random
import numpy as np
import argparse
import torchvision
import time
import tqdm
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset.dataset import CustomDataset
from .utils.uda import target_loss, masking, slide_pred
from .utils.eval import accuracy, miou
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

def set_seed(seed):

    if seed != -1:
        print(f"Seed:{seed}")
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)  # type: ignore
        torch.backends.cudnn.deterministic = True  # type: ignore
        torch.backends.cudnn.benchmark = True  # type: ignore



def train_begin(training,loader,running_loss,running_acc,running_iou):
    if training :
        desc = "Train"
    else:
        desc = "Valid"

    progress = tqdm.tqdm(loader,desc=f'Epoch:{epoch+1}/{args.epochs}')
    for i,data in enumerate(progress):

        source_img,source_mask,target_img,distortion_img,distortion_mask = data

        source_img = source_img.to(model.device)
        source_mask = source_mask.to(model.device).long()

        loss_mask = torch.tensor(0.0).to(model.device)
        loss_distortion = torch.tensor(0.0).to(model.device)


        if (training):
            optimizer.zero_grad()


            target_img = target_img.to(model.device)
            distortion_img = distortion_img.to(model.device)
            distortion_mask = distortion_mask.to(model.device).long()

            with torch.no_grad():
                pseudo_label = teacher_model(target_img)['out']
                pseudo_label = torch.softmax(pseudo_label, dim=1) # b,c,h,w


            #DACS 기법 : 이미지별 일부 클래스 섞기 ( 학습 에폭이 큰 경우에만 잘 작동하는 것으로 보이기에 삭제 )
            '''mixed_target_img,mixed_pseudo_label = dacs(source_img,source_mask,target_img,pseudo_label,mix_ratio)

            pred_mixed_target = model(mixed_target_img)['out']

            loss_mix = LT(pred_mixed_target,mixed_pseudo_label) * lambda_mix'''


            #Distortion loss
            pred_distortion = model(distortion_img)['out']

            loss_distortion = LS(pred_distortion,distortion_mask)


            # MIC 기법 : 패치 일부 masking
            target_img = masking(target_img, args.masking_ratio,args.mask_size)

            pred_target = model(target_img)['out']

            loss_mask = LT(pred_target,pseudo_label) * args.lambda_mask



        pred_source = model(source_img)['out']


        loss_source = LS(pred_source,source_mask)


        loss_total = loss_source + loss_distortion + loss_mask

        acc = accuracy(pred_source,source_mask)

        iou = miou(pred_source,source_mask)


        if (training):

            loss_total.backward()

            optimizer.step()

        running_loss += [loss_source.detach().cpu().numpy(),loss_distortion.detach().cpu().numpy(),loss_mask.detach().cpu().numpy()]
        running_acc += [acc.cpu()]
        running_iou += [iou.cpu()]

        progress.set_description(f'Epoch:{epoch+1}/{args.epochs} | {desc}_Acc:{np.round(running_acc/(i+1),4)} | {desc}_IoU:{np.round(running_iou/(i+1),4)} | {desc}_Loss:{np.round(running_loss/(i+1),4)} | Self-Training:{running_loss[-1]>0}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type="str", default="./data(resized)/")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup_epochs", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=12)

    parser.add_argument("--EMA_alpha", type=float, default=0.9)
    parser.add_argument("--masking_ratio", type=float, default=0.75)
    parser.add_argument("--mask_size", type=int, default=32)
    parser.add_argument("--lambda_mask", type=int, default=1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type="str", default="./ckpt/")
    parser.add_argument("--checkpoint", type=str, default="")
    
    args = parser.parse_args()

    device = "cuda"

    set_seed(args.seed)
    train_set = CustomDataset(csv_file=[f'./train_source.csv', f'./val_source.csv', f'./val_source.csv'],
                              data_path=args.data_path,
                              target_csv_file = f'./train_target.csv',
                              mix_bg_prob = 0.25
                              )
    valid_set = CustomDataset(csv_file=f'./val_source.csv',
                              data_path=args.data_path
                              ) #실제로 사용하지는 않음, 1/10의 데이터를 사용하여도 100에폭까지 오버피팅이 발생하지 않았음

    
    learning_status = {
        'train_accs' : [],
        'valid_accs' : [],
        'train_ious' : [],
        'valid_ious' : [],
        'train_losses' : [],
        'valid_losses' : [],
        'lrs' : []
    }
    min_epoch = 0

    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 1e-3}], weight_decay=0.01)
    
    save_last = True

    # Warmup 파라미터
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,last_epoch=len(learning_status['lrs'])-1, lr_lambda=lambda epoch: epoch / args.warmup_epochs)

    if args.checkpoint != "":
        learning_status = pd.read_csv(os.path.join(args.checkpoint.dirname, 'status.csv'),index_col=0).to_dict(orient='list')
        checkpoint = torch.load(args.checkpoint, device)

        model.load_state_dict(checkpoint['model'])
        teacher_model.load_state_dict(checkpoint['teacher_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

        for k in learning_status.keys():
            for i in range(len(learning_status[k])):
                if k!='lrs':
                    learning_status[k][i] = np.array([item for item in learning_status[k][i][1:-1].split(' ') if item != ''],dtype=np.float32)

        learning_status = {
            'train_accs': learning_status['train_accs'][:checkpoint['epoch']],
            'valid_accs': learning_status['valid_accs'][:checkpoint['epoch']],
            'train_ious': learning_status['train_ious'][:checkpoint['epoch']],
            'valid_ious': learning_status['valid_ious'][:checkpoint['epoch']],
            'train_losses': learning_status['train_losses'][:checkpoint['epoch']],
            'valid_losses': learning_status['valid_losses'][:checkpoint['epoch']],
            'lrs': learning_status['lrs'][:checkpoint['epoch']]
        }

        min_epoch = np.argmin(np.sum(learning_status['valid_losses'],-1))

        

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model_save_path = os.path.join(args.save_path, f"exp_{os.listdir(args.save_path):02d}")

    model = model.to(device)
    teacher_model = teacher_model.to(device)
    LS = nn.CrossEntropyLoss()
    LT = target_loss(t=0.968)

    print("모델 저장 경로 : "+ model_save_path)
    fit_time = time.time()
    start_epoch = len(learning_status['lrs'])
    teacher_model.eval()


    for i in range(len(learning_status['valid_accs'])):
        print(f"LR : {learning_status['lrs'][i]}")
        print(f"Epoch:{i+1}/{args.epochs} | Train_Acc : {np.round(learning_status['train_accs'][i],4)} | Train_IoU : {np.round(learning_status['train_ious'][i],4)} | Train_Loss : {np.round(learning_status['train_losses'][i],4)}")
        print(f"Epoch:{i+1}/{args.epochs} | Valid_Acc : {np.round(learning_status['valid_accs'][i],4)} | Valid_IoU : {np.round(learning_status['valid_ious'][i],4)} | Valid_Loss : {np.round(learning_status['valid_losses'][i],4)}")
        print()

    for epoch in range(start_epoch,args.epochs):

        #Warmup Schedule
        if epoch < args.warmup_epochs :
            scheduler.step()
            print("lr이 변경되었습니다.",optimizer.param_groups[0]['lr'])

        #EMA Update
        alpha_teacher = min(1 - 1 / (epoch + 1), args.EMA_alpha)
        for ema_param, param in zip(teacher_model.parameters(), model.parameters()):
            ema_param.data = alpha_teacher * ema_param.data + (1 - alpha_teacher) * param.data

        print("EMA Weight Update 적용, Alpha =",alpha_teacher)

        running_train_loss = np.array([0.0,0.0,0.0])
        running_valid_loss = np.array([0.0,0.0,0.0])

        running_train_acc = np.array([0.0])
        running_valid_acc = np.array([0.0])

        running_train_iou = np.array([0.0])
        running_valid_iou = np.array([0.0])

        model.train()
        train_begin(True,train_loader,running_train_loss,running_train_acc,running_train_iou)
        model.eval()
        # Validation 코드
        # with torch.no_grad():
        #     train_begin(False,valid_loader,running_valid_loss,running_valid_acc,running_valid_iou)
        
        os.makedirs(model_save_path,exist_ok=True)

        learning_status['train_losses'].append((running_train_loss/len(train_loader)))
        learning_status['valid_losses'].append((running_valid_loss/len(valid_loader)))
        learning_status['train_accs'].append((running_train_acc/len(train_loader)))
        learning_status['valid_accs'].append((running_valid_acc/len(valid_loader)))
        learning_status['train_ious'].append((running_train_iou/len(train_loader)))
        learning_status['valid_ious'].append((running_valid_iou/len(valid_loader)))
        learning_status['lrs'].append(optimizer.param_groups[0]['lr'])

        df = pd.DataFrame(learning_status)

        checkpoint = {
            'epoch': epoch+1 , #에폭
            'model': model.state_dict(),  # 모델
            'teacher_model': teacher_model.state_dict(), # Teacher 모델
            'optimizer': optimizer.state_dict(),  # 옵티마이저
            'scheduler': scheduler.state_dict(),  # 스케줄러
        }

        torch.save(checkpoint, os.path.join(model_save_path,"Last.pth"))
        if (epoch+1)%5 == 0:
            torch.save(checkpoint, os.path.join(model_save_path, f'Epoch({epoch+1:03d}).pth'))

        df.to_csv(os.path.join(model_save_path, 'status.csv'), index=True)


        if sum(learning_status['valid_losses'][min_epoch]) >= sum(learning_status['valid_losses'][-1]) and sum(learning_status['valid_losses'][-1] > 0):
            print(f"Valid Loss가 최소가 됐습니다. ({sum(learning_status['valid_losses'][min_epoch]):.4f}({min_epoch+1}) -> {sum(learning_status['valid_losses'][-1]):.4f}({len(learning_status['valid_losses'])}))")
            print(f'해당 모델이 {model_save_path}Best.pth 경로에 저장됩니다.')
            min_epoch = len(learning_status['valid_losses'])-1
            torch.save(checkpoint, os.path.join(model_save_path, 'Best.pth'))
        else:
            print(f"Valid_Loss가 최소가 되지 못했습니다.(최소 Epoch:{min_epoch+1} : {sum(learning_status['valid_losses'][min_epoch]):.4f})")

        print('')

    print('학습 최종 시간: {:.2f} 분\n' .format((time.time()- fit_time)/60))
