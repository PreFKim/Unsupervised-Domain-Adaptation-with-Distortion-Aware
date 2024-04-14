import json, tqdm, os, cv2, glob
import numpy as np
import argparse

def save_class_stats(out_dir, sample_class_stats):
    sample_class_stats = [e for e in sample_class_stats if e is not None]
    with open(os.path.join(out_dir, 'sample_class_stats.json'), 'w') as of:
        json.dump(sample_class_stats, of, indent=2)

    sample_class_stats_dict = {} #파일별 각 클래스의 픽셀 수 합
    for stats in sample_class_stats:
        f = stats.pop('file')
        sample_class_stats_dict[f] = stats
    with open(os.path.join(out_dir, 'sample_class_stats_dict.json'), 'w') as of:
        json.dump(sample_class_stats_dict, of, indent=2)

    samples_with_class = {} #각 클래스별 파일 내의 클래스 수
    for file, stats in sample_class_stats_dict.items():
        for c, n in stats.items():
            if c not in samples_with_class:
                samples_with_class[c] = [(file, n)]
            else:
                samples_with_class[c].append((file, n))
    with open(os.path.join(out_dir, 'samples_with_class.json'), 'w') as of:
        json.dump(samples_with_class, of, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/samsung_seg")
    args = parser.parse_args()

    root_path = args.data_path

    sample_class_stats = []
    for filename in tqdm.tqdm(sorted(glob.glob(f'{root_path}/*_gt/*.png'))):
        mask = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)

        mask[mask==255] = 12

        summation_classes_per_file = {}
        for i in range(13):
            n = int(np.sum(mask==i))
            if n>0:
                summation_classes_per_file[int(i)]=n
        summation_classes_per_file['file'] = filename
        sample_class_stats.append(summation_classes_per_file)

    save_class_stats(root_path, sample_class_stats)