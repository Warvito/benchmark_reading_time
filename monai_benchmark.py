# Code based on https://github.com/Project-MONAI/tutorials/blob/master/acceleration/dataset_type_performance.ipynb
import datetime
import glob
import os
import pathlib
import shutil
import tempfile
import time

import matplotlib.pyplot as plt
import torch
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import CacheDataset, Dataset, PersistentDataset, list_data_collate
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import compute_meandice
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.transforms import (
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadNiftid,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
)
from monai.utils import set_determinism

print_config()

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)


def train_process(train_ds, val_ds):
    # use batch_size=2 to load images and use RandCropByPosNegLabeld
    # to generate 2 x 4 images for network training
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=2, shuffle=True, num_workers=4, collate_fn=list_data_collate,
    )
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, num_workers=4)
    device = torch.device("cuda:0")
    model = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)

    epoch_num = 600
    val_interval = 1  # do validation for every epoch
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    epoch_times = list()
    total_start = time.time()
    for epoch in range(epoch_num):
        epoch_start = time.time()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epoch_num}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step_start = time.time()
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(
                f"{step}/{len(train_ds) // train_loader.batch_size}, train_loss: {loss.item():.4f}"
                f" step time: {(time.time() - step_start):.4f}"
            )
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                metric_sum = 0.0
                metric_count = 0
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    roi_size = (160, 160, 160)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(
                        val_inputs, roi_size, sw_batch_size, model
                    )
                    value = compute_meandice(
                        y_pred=val_outputs,
                        y=val_labels,
                        include_background=False,
                        to_onehot_y=True,
                        mutually_exclusive=True,
                    )
                    metric_count += len(value)
                    metric_sum += value.sum().item()
                metric = metric_sum / metric_count
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(
                        model.state_dict(), os.path.join(root_dir, "best_metric_model.pth"),
                    )
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f" best mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}"
                )
        print(f"time of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
        epoch_times.append(time.time() - epoch_start)

    print(
        f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
        f" total time: {(time.time() - total_start):.4f}"
    )
    return (
        epoch_num,
        time.time() - total_start,
        epoch_loss_values,
        metric_values,
        epoch_times,
    )


resource = "https://drive.google.com/uc?id=1jzeNU1EKnK81PyTsrx0ujfNl-t0Jo8uE"
md5 = "410d4a301da4e5b2f6f86ec3ddba524e"

compressed_file = os.path.join(root_dir, "Task09_Spleen.tar")
data_dir = os.path.join(root_dir, "Task09_Spleen")
if not os.path.exists(data_dir):
    download_and_extract(resource, compressed_file, root_dir, md5)

train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
data_dicts = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(train_images, train_labels)
]
train_files, val_files = data_dicts[:-9], data_dicts[-9:]


def transformations():
    train_transforms = Compose(
        [
            LoadNiftid(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Spacingd(
                keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest"),
            ),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            # randomly crop out patch samples from big image based on pos / neg ratio
            # the image centers of negative samples must be in valid image area
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            ToTensord(keys=["image", "label"]),
        ]
    )

    # NOTE: No random cropping in the validation data, we will evaluate the entire image using a sliding window.
    val_transforms = Compose(
        [
            LoadNiftid(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Spacingd(
                keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest"),
            ),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            ToTensord(keys=["image", "label"]),
        ]
    )
    return train_transforms, val_transforms


set_determinism(seed=0)
train_trans, val_trans = transformations()
train_ds = Dataset(data=train_files, transform=train_trans)
val_ds = Dataset(data=val_files, transform=val_trans)

(epoch_num, total_time, epoch_loss_values, metric_values, epoch_times,) = train_process(
    train_ds, val_ds
)
print(f"total training time of {epoch_num} epochs with regular Dataset: {total_time:.4f}")

persistent_cache = pathlib.Path(root_dir, "persistent_cache")
persistent_cache.mkdir(parents=True, exist_ok=True)

set_determinism(seed=0)
train_trans, val_trans = transformations()
train_persitence_ds = PersistentDataset(
    data=train_files, transform=train_trans, cache_dir=persistent_cache
)
val_persitence_ds = PersistentDataset(
    data=val_files, transform=val_trans, cache_dir=persistent_cache
)

(
    persistence_epoch_num,
    persistence_total_time,
    persistence_epoch_loss_values,
    persistence_metric_values,
    persistence_epoch_times,
) = train_process(train_persitence_ds, val_persitence_ds)
print(
    f"total training time of {persistence_epoch_num}"
    f" epochs with persistent storage Dataset: {persistence_total_time:.4f}"
)

set_determinism(seed=0)
train_trans, val_trans = transformations()
cache_init_start = time.time()
cache_train_ds = CacheDataset(
    data=train_files, transform=train_trans, cache_rate=1.0, num_workers=4
)
cache_val_ds = CacheDataset(data=val_files, transform=val_trans, cache_rate=1.0, num_workers=4)
cache_init_time = time.time() - cache_init_start

(
    cache_epoch_num,
    cache_total_time,
    cache_epoch_loss_values,
    cache_metric_values,
    cache_epoch_times,
) = train_process(cache_train_ds, cache_val_ds)
print(f"total training time of {cache_epoch_num} epochs with CacheDataset: {cache_total_time:.4f}")

set_determinism(seed=0)
train_trans, val_trans = transformations()
cache_init_start = time.time()
cache_train_ds = CacheDataset(
    data=train_files, transform=train_trans, cache_rate=1.0, num_workers=4
)
cache_val_ds = CacheDataset(data=val_files, transform=val_trans, cache_rate=1.0, num_workers=4)
cache_init_time = time.time() - cache_init_start

(
    cache_epoch_num,
    cache_total_time,
    cache_epoch_loss_values,
    cache_metric_values,
    cache_epoch_times,
) = train_process(cache_train_ds, cache_val_ds)
print(f"total training time of {cache_epoch_num} epochs with CacheDataset: {cache_total_time:.4f}")

plt.figure("train", (12, 6))

plt.subplot(1, 2, 1)
plt.title("Total Train Time(600 epochs)")
plt.bar("regular", total_time, 1, label="Regular Dataset", color="red")
plt.bar(
    "persistent", persistence_total_time, 1, label="Persistent Dataset", color="blue",
)
plt.bar(
    "cache", cache_init_time + cache_total_time, 1, label="Cache Dataset", color="green",
)
plt.bar("cache", cache_init_time, 1, label="Cache Init", color="orange")
plt.ylabel("secs")
plt.grid(alpha=0.4, linestyle=":")
plt.legend(loc="best")

plt.subplot(1, 2, 2)
plt.title("Epoch Time")
x = [i + 1 for i in range(len(epoch_times))]
plt.xlabel("epoch")
plt.ylabel("secs")
plt.plot(x, epoch_times, label="Regular Dataset", color="red")
plt.plot(x, persistence_epoch_times, label="Persistent Dataset", color="blue")
plt.plot(x, cache_epoch_times, label="Cache Dataset", color="green")
plt.grid(alpha=0.4, linestyle=":")
plt.legend(loc="best")

plt.savefig(f"/benchmark/results_{timestamp}.png")

if directory is None:
    shutil.rmtree(root_dir)
