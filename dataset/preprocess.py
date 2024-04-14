import argparse, tqdm, cv2, os
import numpy as np

from ..config import IMAGE_HEIGHT, IMAGE_WIDTH

def image_preprocess(root_path,save_path):
    dirs = ['test_image','train_source_image','train_target_image','val_source_image']

    if os.path.exists(save_path) == False:
        os.makedirs(save_path)

    for dir in dirs:
        path = os.path.join(root_path, dir)
        for filename in tqdm.tqdm(os.listdir(path) , desc = dir):
            write_path = os.path.join(save_path,filename)

            img = cv2.imread(filename)
            img = cv2.resize(img,(IMAGE_WIDTH,IMAGE_HEIGHT))
            cv2.imwrite(write_path,img)

def gt_preprocess(root_path):
    dirs = ['train_source_gt','val_source_gt']

    if os.path.exists(save_path) == False:
        os.makedirs(save_path)
    
    for dir in dirs:
        path = os.path.join(root_path, dir)
        for filename in tqdm.tqdm(os.listdir(path) , desc = dir):
            write_path = os.path.join(save_path,filename)

            img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
            img[img==255] = 12

            n_values = np.max(img) + 1
            img = np.eye(n_values)[img]

            img = cv2.resize(img,(IMAGE_WIDTH,IMAGE_HEIGHT),cv2.INTER_LINEAR)

            img = np.argmax(img,-1)

            img[img==12] = 255
            cv2.imwrite(write_path,img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/")
    args = parser.parse_args()

    root_path = args.data_path
    if root_path[-1] == '/':
        root_path = root_path[:-1]
    save_path = root_path + "(resized)"

    print(f"Root path : {root_path}")
    print(f"Save path : {save_path}")

    image_preprocess(root_path,save_path)
    gt_preprocess(root_path,save_path)
    
