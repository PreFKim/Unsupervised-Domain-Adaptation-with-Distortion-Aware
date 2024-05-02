from torch.utils.data import Dataset
import pandas as pd
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import *

class CustomDataset(Dataset):
    def __init__(self, csv_file, data_path, target_csv_file=None, mix_bg_prob = 0, infer=False):

        #여러 csv 파일 합치기
        if type(csv_file) == list:
            self.data = pd.read_csv(csv_file[0])
            for i in range(1,len(csv_file)):
                self.data = pd.concat([self.data, pd.read_csv(csv_file[i])], ignore_index=True)
        else :
            self.data = pd.read_csv(csv_file)

        self.infer = infer

        self.mix_bg_prob = mix_bg_prob
        self.data_path = data_path

        #공통으로 적용되는 Transform

        self.augmentation = A.Compose([
                A.ColorJitter(p=0.25),
                A.HorizontalFlip(p=0.5),
                A.RandomCrop(width = CROP_WIDTH, height = CROP_HEIGHT)
            ])

        self.source_norm = A.Normalize(mean = source_mean , std = source_std)
        self.target_norm = A.Normalize(mean = target_mean , std = target_std)

        self.len_source = len(self.data)

        if target_csv_file :
            self.target_data = pd.read_csv(target_csv_file)
            self.len_target = len(self.target_data)

        else :
            self.target_data = None
            self.len_target = 1

    def mix_bg(self,source_image,source_mask,target_image):

        h,w,c = source_image.shape

        # 타원의 중심 좌표와 크기 설정 (예: 중심 (x, y), 장축 반지름 a, 단축 반지름 b)
        center = (IMAGE_WIDTH//2, int(IMAGE_HEIGHT*0.375))  # x,y
        axis_length = (IMAGE_WIDTH//2, int(IMAGE_HEIGHT*0.64))  # 장축 반지름과 단축 반지름

        # 타원 그리기 (타원을 1로 채우고 나머지 부분은 0으로 채움)
        mask = np.zeros((h,w,1),dtype=np.uint8)
        cv2.ellipse(mask, center, axis_length, 0, 0, 360, 1, -1)

        # 타원 모양으로 crop된 이미지 생성
        mixed_image = source_image * mask + target_image * (1-mask)
        mixed_mask = source_mask * mask[:,:,0] + np.ones_like(source_mask)*12*(1-mask[:,:,0])

        return mixed_image, mixed_mask

    def __len__(self):
        return self.len_source

    def __getitem__(self, idx):
        source_idx = idx

        img_path = self.data.iloc[source_idx, 1].replace('./',self.data_path+'/')
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.infer == True:
            image = A.Compose([self.target_norm,
                               ToTensorV2()])(image=image)['image']
            return image


        mask_path = self.data.iloc[source_idx, 2].replace('./',self.data_path+'/')
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask[mask == 255] = 12

        # 보닛 부분에 있는 하늘 클래스들을 배경클래스로 변환
        mask[IMAGE_HEIGHT//3*2:][mask[IMAGE_HEIGHT//3*2:]==8] = 12

        target_image = np.zeros_like(image) 
        distortion_image = image.copy()
        distortion_mask = mask.copy()

        if self.target_data is not None:
            target_img_path = self.target_data.iloc[idx % self.len_target, 1].replace('./',self.data_path+'/')
            target_image = cv2.imread(target_img_path)
            target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)

            if self.mix_bg_prob > np.random.uniform(0,1):
                distortion_image,distortion_mask = self.mix_bg(distortion_image.astype(np.uint8),distortion_mask.astype(np.uint8),target_image)

        source_tensor = A.Compose([self.augmentation,
                                   self.source_norm,
                                   ToTensorV2()])(image=image,mask=mask)

        target_image = A.Compose([self.augmentation,
                                  self.target_norm,
                                  ToTensorV2()])(image=target_image)['image']

        distortion_tensor = A.Compose([self.augmentation,
                                    A.ElasticTransform(alpha=100, sigma=10, alpha_affine=25,border_mode = 1,p=1),
                                    self.source_norm,
                                    ToTensorV2()])(image=distortion_image,mask= distortion_mask)

        image = source_tensor['image']
        mask = source_tensor['mask']
        distortion_image = distortion_tensor['image']
        distortion_mask = distortion_tensor['mask']


        return image, mask, target_image, distortion_image, distortion_mask