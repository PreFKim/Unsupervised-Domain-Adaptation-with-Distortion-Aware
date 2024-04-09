#사전에 미리 Resize해둔 이미지 크기
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 512

#Crop한 이미지의 크기
CROP_WIDTH = IMAGE_WIDTH//2
CROP_HEIGHT = IMAGE_HEIGHT//2

#Train_Source + Valid_Source 의 Mean, Std
source_mean = [0.5897106 , 0.5952661 , 0.57897425]
source_std = [0.16688786, 0.15721062, 0.1589595]

#Train_Target 의 Mean, Std
target_mean = [0.4714665 , 0.47141412, 0.49733913]
target_std = [0.23908237, 0.24033973, 0.25281718]

palette = [
    [0,0,0],
    [0,255,0],
    [127,127,127],
    [255,0,255],
    [153,76,0],
    [0,153,153],
    [255,0,0],
    [204,255,204],
    [0,255,255],
    [255,255,204],
    [255,0,127],
    [255,127,0],
    [255,255,255]
    ]

cls_name = [
    'Road',
    'Sidewalk',
    'Construction',
    'Fence',
    'Pole',
    'Traffic Light',
    'Traffic Sign',
    'Nature',
    'Sky',
    'Person',
    'Rider',
    'Car',
    'None'
    ]