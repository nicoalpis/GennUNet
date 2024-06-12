import multiprocessing
import monai
import numpy as np
import logging
import os
import argparse
import pandas as pd
import json

def calculate_organ_volumes(label):
    # Assuming the voxel dimensions are in mm
    voxel_volume_mm3 = np.prod([1.5, 1.5, 1.5])

    # Calculate the volume for each organ
    organ_volumes = []
    # unique_organs = np.unique(label)
    for organ in range(13):
        if organ == 0:  # Assuming 0 is the background
            continue
        organ_volume = np.sum(label == organ) * voxel_volume_mm3
        organ_volumes.append(organ_volume)

    return organ_volumes

def run(train_files):
    transforms = monai.transforms.Compose(
        [
        monai.transforms.LoadImaged(keys=["image", "label"], image_only=False),
        monai.transforms.EnsureChannelFirstd(keys=["image", "label"]),
        monai.transforms.Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 1.5),
            mode=("bilinear", "nearest"),
        ),
        ]
    )

    results = transforms(train_files)
    label = results['label']

    volumes = calculate_organ_volumes(label)

    return volumes

if __name__ ==  '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--fold', type=int, required=True)

    args = parser.parse_args()

    data_dir = "/data/Dataset060_Merged_Def/"
    plans_dir = (
        "/data/Dataset060_Merged_Def/"
    )

    with open(os.path.join(plans_dir, "splits_Dataset060_Merged_Def.json"), "r") as file:
        plans = json.load(file)

    train_images = []
    train_labels = []
    val_images = []
    val_labels = []

    for image_name in plans[args.fold]["train"]:
        train_images.append(os.path.join(data_dir, "imagesTr", image_name + "_0000.nii.gz"))
        train_labels.append(os.path.join(data_dir, "labelsTr", image_name + ".nii.gz"))

    train_images = sorted(train_images)
    train_labels = sorted(train_labels)

    train_files = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(train_images, train_labels)
    ]

    with multiprocessing.Pool(processes=7) as pool:
        result = pool.map(run, train_files)

    np.save(f'/organ_volumes_per_fold/fold_{args.fold}.npy', np.array(result))