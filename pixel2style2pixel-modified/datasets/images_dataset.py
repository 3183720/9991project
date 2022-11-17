import json

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from utils import data_utils


class ImagesDataset(Dataset):
    def __init__(
        self,
        source_root,
        target_root,
        opts,
        latents_root=None,
        labels_path=None,
        target_transform=None,
        source_transform=None,
    ):

        self.target_paths = sorted(data_utils.make_dataset(target_root))
        self.source_paths = sorted(data_utils.make_dataset(source_root))
        self.latent_paths = None
        if latents_root is not None:
            self.latent_paths = sorted(
                data_utils.make_latents_dataset(latents_root)
            )

        self.path_to_label = None
        if labels_path is not None:
            with open(labels_path) as f:
                labels = json.load(f)["labels"]
            self.path_to_label = {path: label for path, label in labels}

        self.source_transform = source_transform
        self.target_transform = target_transform
        self.opts = opts

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        from_path = self.source_paths[index]
        from_im = Image.open(from_path)
        from_im = (
            from_im.convert("RGB")
            if self.opts.label_nc == 0
            else from_im.convert("L")
        )

        to_path = self.target_paths[index]
        to_im = Image.open(to_path).convert("RGB")

        if self.target_transform:
            to_im = self.target_transform(to_im)
        if self.source_transform:
            from_im = self.source_transform(from_im)
        else:
            from_im = to_im
        latent = None

        if self.path_to_label is not None:
            #label_list = self.path_to_label[from_path]
            #print(from_path)
            label = from_path.split('/')[3]
            label= label.split('_')[0]
            #label = label[:, label_list]
            return from_im, to_im, int(label)

        if self.latent_paths is not None:
            latent_path = self.latent_paths[index]
            z = np.load(latent_path)
            latent = torch.from_numpy(z)

            return from_im, to_im, latent

        return from_im, to_im
