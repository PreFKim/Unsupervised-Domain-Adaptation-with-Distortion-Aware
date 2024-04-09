# Unsupervised-Domain-Adaptation-with-Distortion-Aware
2023 Samsung AI Challenge : Camera-Invariant Domain Adaptation (2023-08-21 ~ 2023-10-02) [Private 9th Solution]

대회 사이트 주소 : [DACON, 2023 Samsung AI Challenge : Camera-Invariant Domain Adaptation](https://dacon.io/competitions/official/236132/overview/description)

---

## 대회 설명

왜곡이 존재하지 않는 이미지(Source Domain)와 레이블을 활용하여, 왜곡된 이미지(Target Domain)에 대해서도 고성능의 이미지 분할을 수행하도록 하는 Unsupervised Domain Adaptation(UDA) 알고리즘을 개발하는 대회

## UDA 구조

<p align="center"><img src="./img/UDA.PNG"></p>

---

## 추론 방식

<p align="center"><img src="./img/preprocess.PNG"></p>

<p align="center"><img src="./img/postprocess.PNG"></p>

### MiT-B3 기준 FPS

GPU : Tesla T4 

|Image Size|Crop Size|Stride|FPS|
|---|---|---|---|
|(256, 512)|X|X|27FPS|
|(512, 1024)|X|X|10FPS|
|(512, 1024)|(256, 512)|(128, 256)|25FPS|

---

## 성능

### Crop크기에 따른 성능차이

|Image Size|Crop Size|Public Score|
|---|---|---|
|(256, 512)|X|0.5655|
|(256, 512)|(128, 256)|0.5517|
|(512, 1024)|(256, 512)|0.6258|

### Encoder에 따른 성능과 FPS

|Encoder|FPS|Public Score|Weight|
|---|---|---|---|
|MiT-B3|25|0.6231|[weights](https://drive.google.com/file/d/1vzQOudyrv-0gtI1pWJ-WoQaRZWC6agWv/view?usp=drive_link)|
|MiT-B4|15|0.6258|[weights](https://drive.google.com/file/d/1yBU_3mqMkzyvOcs_skZOQlQXvSg5kBEs/view?usp=drive_link)|
|MiT-B5|10|0.6099|[weights](https://drive.google.com/file/d/1lIR4q2hWVcLGyAHskTxa6j9VqK27S1PN/view?usp=drive_link)|



---

## Environment

- 사용 환경 : Colab Pro Plus

- OS : Ubuntu 22.04.2 LTS

- GPU : A100 40GB

- Cuda 버전 : 11.8

- Python 버전 : 3.10.12 (main, Jun 11 2023, 05:26:28) [GCC 11.4.0]

- 라이브러리 버전
    - numpy: 1.23.5
    - matplotlib: 3.7.1
    - pandas: 1.5.3
    - tqdm: 4.66.1
    - torch: 2.0.1
    - torchvision: 0.15.2
    - albumentations: 1.3.1
    - cv2: 4.8.0
    - timm: 0.9.7
    - huggingface-hub: 0.17.3
    - safetensors: 0.3.3

## How to Use

1. git을 클론한다.

    ```
    git clone https://github.com/PreFKim/Unsupervised-Domain-Adaptation-with-Distortion-Aware.git
    cd Unsupervised-Domain-Adaptation-with-Distortion-Aware
    git clone https://github.com/VainF/DeepLabV3Plus-Pytorch.git /content/Deeplabv3
    ```

2. [DACON, 2023 Samsung AI Challenge : Camera-Invariant Domain Adaptation](https://dacon.io/competitions/official/236132/overview/description)에서 데이터셋을 다운로드 받는다.

3. 아래의 디렉토리 구조와 동일하게 세팅한다.

4. config.py에서 이미지의 크기를 지정한다.

5. preprocess.py 를 실행시켜 데이터를 전처리한다.

    ```
    python ./dataset/preprocess.py
    ```

6. train.py 코드 내에서 모델을 선택하고 학습을 진행한다.
    ```
    python ./train.py
    ```

7. inference.py 코드 내에서 모델을 선택하고 학습을 진행한다.
    ```
    python ./inference.py
    ```


## Directory 구조

```
    Unsupervised-Domain-Adaptation-with-Distortion-Aware
    ├─ data
    │   ├─ test_image
    │   ├─ train_source_gt
    │   ├─ train_source_image
    │   ├─ train_target_image
    │   ├─ val_source_gt
    │   └─ val_source_image
    ├─  img
    │   └─  ...
    ├─  model
    │   └─  ...
    │   pretrain
    │   └─  segformer.b4.1024x1024.city.160k.pth
    ├─  utils
    │   └─  ...
    ├─  baseline_submit.csv
    ├─  config.py
    ├─  inference.py
    ├─  README.md
    ├─  sample_submission.csv
    ├─  Samsung AI Challange-DA.pdf
    ├─  train_source.csv
    ├─  train_target.csv
    ├─  train.py
    └─  val_source

```

