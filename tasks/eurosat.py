from __future__ import print_function

import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchgeo.datasets import EuroSAT

from dataloader import *


DATA_ROOT = "./data"


##############################################################
# Model
##############################################################

class Net(nn.Module):
    """
    Custom CNN for EuroSAT.

    Input:
        3 x 128 x 128 RGB image

    Output:
        10 classes

    The architecture is intentionally kept identical to the RESISC45
    network except for the final classification layer. This makes the
    EuroSAT experiment directly compatible with the same FL pipeline.
    """

    def __init__(self, num_classes=10):
        super(Net, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                256 * 8 * 8,
                512
            ),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(
                512,
                num_classes
            )
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


##############################################################
# Dataset Wrapper
##############################################################

class EuroSATWrapper(Dataset):
    """
    Wrapper around TorchGeo EuroSAT that returns exactly the same
    (image, label) interface expected by the existing RESISC45 FL code.

    We use the RGB Sentinel-2 bands:
        B04 = red
        B03 = green
        B02 = blue

    TorchGeo returns samples as dictionaries. This wrapper converts them
    to:
        image: Tensor [3, 128, 128]
        label: int
    """

    def __init__(
        self,
        root,
        split,
        transform=None
    ):
        self.transform = transform

        # EuroSAT is natively 13-band Sentinel-2 data. We explicitly
        # select the RGB bands B04/B03/B02 so that the rest of the
        # pipeline receives the same 3-channel input shape as RESISC45.
        #
        # download=True is safe for repeated runs: TorchGeo verifies the
        # local dataset first and downloads only when required.
        self.dataset = EuroSAT(
            root=root,
            split=split,
            bands=EuroSAT.BAND_SETS["rgb"],
            download=True
        )

        print(
            f"Building label cache ({split})..."
        )

        self.targets = torch.tensor([
            int(self.dataset[i]["label"])
            for i in range(len(self.dataset))
        ])

        # TorchGeo EuroSAT contains 10 classes.
        # Most recent TorchGeo versions expose `classes`.
        if hasattr(self.dataset, "classes"):
            self.classes = self.dataset.classes
        else:
            self.classes = list(range(10))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        image = sample["image"]
        label = int(sample["label"])

        if self.transform:
            image = self.transform(image)

        return image, label


##############################################################
# Transforms
##############################################################

class ScaleEuroSATRGB:
    """
    TorchGeo returns EuroSAT GeoTIFF values as float tensors containing
    Sentinel-2 pixel values, not uint8 RGB intensities. ConvertImageDtype
    alone would therefore NOT rescale them to [0,1].

    TorchGeo's own EuroSAT visualization scales RGB values by 3000 and
    clips to [0,1], so we use the same conversion before normalization.
    """

    def __call__(self, image):
        image = image.to(torch.float32)
        return torch.clamp(image / 3000.0, 0.0, 1.0)


def get_transform():
    """
    Keep the same model input size and normalization interface as RESISC45.

    EuroSAT RGB images are 64 x 64 by default, so they are rescaled
    from Sentinel-2 pixel values to [0,1] and resized to 128 x 128 to
    preserve compatibility with the existing CNN.
    """

    return transforms.Compose([
        ScaleEuroSATRGB(),

        transforms.Resize(
            (128, 128)
        ),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],
            std=[
                0.229,
                0.224,
                0.225
            ]
        )
    ])


##############################################################
# Dataset
##############################################################

def getDataset():
    dataset = EuroSATWrapper(
        root=DATA_ROOT,
        split="train",
        transform=get_transform()
    )

    return dataset


##############################################################
# Stored Loader
##############################################################

class StoredLoader:
    """
    Recreates a previously saved partition.
    Behaves like customDataLoader.
    """

    def __init__(
        self,
        dataset,
        partition_list,
        bsz=128
    ):
        self.dataset = dataset
        self.partition_list = partition_list
        self.bsz = bsz
        self.size = len(partition_list)

        self.classes = np.unique(
            dataset.targets
        ).tolist()

    def __len__(self):
        return self.size

    def __getitem__(self, rank):
        partition = Partition(
            self.dataset,
            self.partition_list[rank]
        )

        partition.classes = self.classes

        return DataLoader(
            partition,
            batch_size=self.bsz,
            shuffle=True,
            drop_last=True
        )


##############################################################
# FL Loaders
##############################################################

def basic_loader(
    num_clients,
    loader_type
):
    dataset = getDataset()

    return loader_type(
        num_clients,
        dataset
    )


##############################################################
# Partition Utilities
##############################################################

def save_partition(
    loader,
    path
):
    with open(path, "wb") as f:
        pickle.dump(
            loader.partition_list,
            f
        )

    print(
        f"Saved partition to {path}"
    )


def load_partition(
    path,
    batch_size=128
):
    print(
        f"Loading partition from {path}"
    )

    with open(path, "rb") as f:
        partition_list = pickle.load(f)

    dataset = getDataset()

    return StoredLoader(
        dataset,
        partition_list,
        batch_size
    )


def train_dataloader(
    num_clients,
    loader_type='iid',
    store=True,
    batch_size=128
):
    assert loader_type in [
        'iid',
        'byLabel',
        'dirichlet'
    ]

    if loader_type == 'iid':
        loader_fn = iidLoader
    elif loader_type == 'byLabel':
        loader_fn = byLabelLoader
    else:
        loader_fn = dirichletLoader

    os.makedirs(
        "partitions",
        exist_ok=True
    )

    partition_path = os.path.join(
        "partitions",
        f"EuroSAT_"
        f"{loader_type}_"
        f"{num_clients}.pkl"
    )

    ##################################################
    # Load existing partition
    ##################################################

    if store and os.path.exists(
        partition_path
    ):
        return load_partition(
            partition_path,
            batch_size
        )

    ##################################################
    # Create new partition
    ##################################################

    print(
        "Creating new partition..."
    )

    loader = basic_loader(
        num_clients,
        loader_fn
    )

    ##################################################
    # Save partition
    ##################################################

    if store:
        save_partition(
            loader,
            partition_path
        )

    return loader


##############################################################
# Test Loader
##############################################################

def test_dataloader(
    test_batch_size
):
    dataset = EuroSATWrapper(
        root=DATA_ROOT,
        split="test",
        transform=get_transform()
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=test_batch_size,
        shuffle=False
    )


##############################################################
# Main
##############################################################

if __name__ == '__main__':
    from torchsummary import summary

    print(
        "# Initialize network"
    )

    net = Net(
        num_classes=10
    )

    summary(
        net.cuda(),
        (3, 128, 128)
    )

    print(
        "\n# Initialize dataloaders"
    )

    loader_types = [
        'iid',
        'byLabel',
        'dirichlet'
    ]

    for loader_type in loader_types:
        loader = train_dataloader(
            10,
            loader_type,
            store=False
        )

        print(
            f"\nLoader type: {loader_type}"
        )

        sizes = [
            len(loader[i].dataset)
            for i in range(
                len(loader)
            )
        ]

        print(
            "Client sizes:",
            sizes
        )

        print(
            "Total samples:",
            sum(sizes)
        )

    x = next(
        iter(loader[0])
    )[0].cuda()

    y = net(x)

    print(
        "\nInput:",
        x.shape
    )

    print(
        "Output:",
        y.shape
    )
