import os
import argparse

from tqdm import tqdm
import json

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    CropForegroundd,
    RandAffined,
    Compose,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    EnsureChannelFirstd,
    SpatialPadd,
    ToTensord,
    RandCropByPosNegLabeld,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandFlipd,
    RandGaussianSmoothd,
    RandRotated,
)
import monai
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR

from monai.data import (
    DataLoader,
    CacheDataset,
    decollate_batch,
    set_track_meta,
)

import torch
import numpy as np
from time import sleep

from datetime import datetime


def custom_print(*args, **kwargs):
    # Get the current datetime
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Construct the message with the datetime
    formatted_message = f"{current_time} - {' '.join(map(str, args))}"
    # Use the built-in print function to display the message
    print(formatted_message, **kwargs)


# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--weights", type=str, default="/home/nalvarez/experiments_Swin-UNETR/model_swinvit.pt"
)
parser.add_argument("--patch_size", nargs="+", type=int, default=[256, 160, 224])
parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--num_samples_per_image", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--fold", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--name", type=str, default="swin-unetr")
parser.add_argument("--val_interval", type=int, default=1)
parser.add_argument("-c", action="store_true")

args = vars(parser.parse_args())

seed = args["seed"]
np.random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

output_dir = f"/experiments_Swin-UNETR/fold_{args['fold']}/output"

num_samples = args["num_samples_per_image"]

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 1.5),
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-982,
            a_max=1094,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=args["patch_size"]),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=args["patch_size"],
            pos=2,
            neg=1,
            num_samples=args["num_samples_per_image"],
            image_key="image",
            image_threshold=0,
        ),
        RandRotated(
            keys=["image", "label"],
            prob=0.2,
            range_x=(-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi),
            range_y=(-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi),
            range_z=(-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi),
            mode=["bilinear", "nearest"],
        ),
        RandAffined(keys=["image", "label"], scale_range=(0.7, 1.4), prob=0.2, mode=["bilinear", "nearest"]),
        RandGaussianNoised(keys="image", prob=0.1, mean=0, std=0.1),
        RandGaussianSmoothd(keys="image", prob=0.1, sigma_x=(0.5, 1)),
        RandAdjustContrastd(keys="image", prob=0.15, gamma=(0.75, 1.25)),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0]),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[1]),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[2]),
        ToTensord(keys=["image", "label"]),
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 1.5),
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-982,
            a_max=1094,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=args["patch_size"]),
        ToTensord(keys=["image", "label"]),
    ]
)

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

for image_name in plans[args["fold"]]["train"]:
    train_images.append(os.path.join(data_dir, "imagesTr", image_name + "_0000.nii.gz"))
    train_labels.append(os.path.join(data_dir, "labelsTr", image_name + ".nii.gz"))

for image_name in plans[args["fold"]]["val"]:
    val_images.append(os.path.join(data_dir, "imagesTr", image_name + "_0000.nii.gz"))
    val_labels.append(os.path.join(data_dir, "labelsTr", image_name + ".nii.gz"))

train_images = sorted(train_images)
train_labels = sorted(train_labels)
val_images = sorted(val_images)
val_labels = sorted(val_labels)

train_files = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(train_images, train_labels)
]
val_files = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(val_images, val_labels)
]

train_ds = CacheDataset(data=train_files, transform=train_transforms, num_workers=7)
train_loader = DataLoader(train_ds, batch_size=args["batch_size"], shuffle=True)
val_ds = CacheDataset(data=val_files, transform=val_transforms, num_workers=7)
val_loader = DataLoader(val_ds, batch_size=args["batch_size"])

set_track_meta(False)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SwinUNETR(
    img_size=args["patch_size"],
    in_channels=1,
    out_channels=13,
    feature_size=48,
    use_checkpoint=True,
).to(device)

weight = torch.load(args["weights"])
model.load_from(weights=weight)

torch.backends.cudnn.benchmark = True
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=3e-5)

def load_checkpoint(checkpoint_file, model, optimizer):
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print(
        f"Checkpoint saved at epoch {checkpoint['epoch']}. Starting training on epoch {checkpoint['epoch'] + 1}"
    )
    return checkpoint["epoch"]


def train(
    model,
    data_in,
    loss,
    optim,
    max_epochs,
    model_dir,
    device,
    name,
    test_interval=1,
    start_epoch=0,
):
    best_metric = -1
    best_metric_epoch = -1
    save_loss_train = []
    save_loss_test = []
    save_metric_train = []
    save_metric_test = []
    save_metric_test_per_class = []
    save_iou_test_per_class = []
    train_loader, test_loader = data_in

    data_classes = [
        "spleen",
        "kidney_right",
        "kidney_left",
        "gallbladder",
        "esophagus",
        "liver",
        "stomach",
        "aorta",
        "inferior_vena_cava",
        "pancreas",
        "adrenal_gland_right",
        "adrenal_gland_left",
    ]

    for epoch in range(start_epoch, max_epochs):
        print("-" * 50)
        custom_print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        train_epoch_loss = 0
        train_step = 0
        epoch_metric_train = 0
        with tqdm(train_loader, unit="batch") as tepoch:
            for batch_data in tepoch:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                tepoch.set_description(f"{current_time} - Epoch {epoch+1}")
                train_step += 1

                volume = batch_data["image"]
                label = batch_data["label"]

                volume, labels = (volume.to(device), label.to(device))

                optim.zero_grad()
                outputs = model(volume)
                train_loss = loss(outputs, labels)

                train_loss.backward()
                optim.step()

                train_epoch_loss += train_loss.item()
                labels_list = decollate_batch(labels)
                labels_convert = [
                    post_label(label_tensor) for label_tensor in labels_list
                ]

                output_list = decollate_batch(outputs)
                output_convert = [
                    post_pred(output_tensor) for output_tensor in output_list
                ]

                dice_metric(y_pred=output_convert, y=labels_convert)
                iou_metric(y_pred=output_convert, y=labels_convert)

                tepoch.set_postfix(
                    loss=train_loss.item(),
                    dice_score=dice_metric.aggregate(reduction="mean").item(),
                )
                sleep(0.01)
            
            torch.save(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                os.path.join(model_dir, name + "_last_checkpoint.pth"),
            )

            print("-" * 20)

            train_epoch_loss /= train_step
            print(f"Epoch_loss: {train_epoch_loss:.4f}")
            save_loss_train.append(train_epoch_loss)
            np.save(os.path.join(model_dir, name + "_loss_train.npy"), save_loss_train)

            epoch_metric_train = dice_metric.aggregate(reduction="mean").item()
            dice_metric.reset()

            print(f"Epoch_metric: {epoch_metric_train:.4f}")

            iou_metric_train = iou_metric.aggregate(reduction="mean").item()
            iou_metric.reset()

            print(f"IoU_metric: {iou_metric_train:.4f}")

            save_metric_train.append(epoch_metric_train)
            np.save(
                os.path.join(model_dir, name + "_metric_train.npy"), save_metric_train
            )

            if (epoch + 1) % test_interval == 0:

                model.eval()
                with torch.no_grad():
                    test_epoch_loss = 0
                    test_metric = 0
                    epoch_metric_test = 0
                    test_step = 0

                    for test_data in tqdm(test_loader):

                        test_step += 1

                        test_volume = test_data["image"]
                        test_label = test_data["label"]

                        test_volume, test_label = (
                            test_volume.to(device),
                            test_label.to(device),
                        )

                        test_outputs = sliding_window_inference(
                            test_volume, args["patch_size"], args['num_samples_per_image'], model, overlap=0.5
                        )

                        test_loss = loss(test_outputs, test_label)
                        test_epoch_loss += test_loss.item()

                        labels_list = decollate_batch(test_label)
                        labels_convert = [
                            post_label(label_tensor) for label_tensor in labels_list
                        ]

                        output_list = decollate_batch(test_outputs)
                        output_convert = [
                            post_pred(output_tensor) for output_tensor in output_list
                        ]

                        dice_metric(y_pred=output_convert, y=labels_convert)
                        iou_metric(y_pred=output_convert, y=labels_convert)

                    test_epoch_loss /= test_step
                    print(f"test_loss_epoch: {test_epoch_loss:.4f}")
                    save_loss_test.append(test_epoch_loss)
                    np.save(
                        os.path.join(model_dir, name + "_loss_test.npy"), save_loss_test
                    )

                    epoch_metric_test = dice_metric.aggregate(reduction="mean").item()

                    print(f"test_dice_epoch: {epoch_metric_test:.4f}")

                    dice_scores = {
                        key: value
                        for key, value in zip(
                            data_classes,
                            dice_metric.aggregate(reduction="mean_batch").tolist(),
                        )
                    }
                    print("test_dice_epoch_per_class:")
                    for key, value in dice_scores.items():
                        print(f"\t{key}: {value}")
                    save_metric_test_per_class.append(
                        np.array(list(dice_scores.values()))
                    )
                    np.save(
                        os.path.join(model_dir, name + "_metric_test_per_class.npy"),
                        save_metric_test_per_class,
                    )

                    iou_metric_test = iou_metric.aggregate(reduction="mean").item()

                    print(f"test_iou_epoch: {iou_metric_test:.4f}")

                    iou_scores = {
                        key: value
                        for key, value in zip(
                            data_classes,
                            iou_metric.aggregate(reduction="mean_batch").tolist(),
                        )
                    }
                    print("test_iou_epoch_per_class:")
                    for key, value in iou_scores.items():
                        print(f"\t{key}: {value}")
                    save_iou_test_per_class.append(np.array(list(iou_scores.values())))
                    np.save(
                        os.path.join(model_dir, name + "_iou_test_per_class.npy"),
                        save_iou_test_per_class,
                    )

                    iou_metric.reset()
                    save_metric_test.append(epoch_metric_test)
                    np.save(
                        os.path.join(model_dir, name + "_metric_test.npy"),
                        save_metric_test,
                    )
                    dice_metric.reset()
                    if epoch_metric_test > best_metric:
                        best_metric = epoch_metric_test
                        best_metric_epoch = epoch + 1
                        torch.save(
                            {
                                "epoch": epoch + 1,
                                "state_dict": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                            },
                            os.path.join(model_dir, name + "_best_metric_model.pth"),
                        )

                    print(
                        f"current epoch: {epoch + 1} current mean dice: {epoch_metric_test:.4f}"
                        f"\nbest mean dice: {best_metric:.4f} "
                        f"at epoch: {best_metric_epoch}"
                    )

    print(
        f"train completed, best_metric: {best_metric:.4f} "
        f"at epoch: {best_metric_epoch}"
    )


post_label = AsDiscrete(to_onehot=13)
post_pred = AsDiscrete(argmax=True, to_onehot=13)
dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
iou_metric = monai.metrics.MeanIoU(
    include_background=False, reduction="mean", get_not_nans=False
)

start_epoch = 0
if args["c"]:
    checkpoint_file = os.path.join(output_dir, args['name'] + "_last_checkpoint.pth")
    start_epoch = load_checkpoint(checkpoint_file, model, optimizer)
train(
    model=model,
    data_in=(train_loader, val_loader),
    loss=loss_function,
    optim=optimizer,
    max_epochs=args["num_epochs"],
    model_dir=output_dir,
    device=device,
    name=args["name"],
    test_interval=args["val_interval"],
    start_epoch=start_epoch,
)
