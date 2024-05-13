from pathlib import Path
from typing import Union

import idsprites as ids
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

from idsprites.infinite_dsprites import TensorFactors

ImgFactorsTup = tuple[torch.Tensor, TensorFactors]


class FileDataset(Dataset[ImgFactorsTup]):
    def __init__(self, path: Union[Path, str], transform=None, target_transform=None):
        self.path = Path(path)
        self.transform = transform
        self.target_transform = target_transform
        factors = np.load(self.path / "factors.npz", allow_pickle=True)
        # Turn dict of lists into list of dicts.
        # Opposite of idsprites.io.Task.write_split
        factors = [
            dict(zip(factors, value)) for value in zip(*factors.values())
        ]
        self.data = [ids.Factors(**factors) for factors in factors]
        self.shapes = np.load(self.path / "../shapes.npy", allow_pickle=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> ImgFactorsTup:
        img_path = self.path / f"sample_{idx}.png"
        image = read_image(str(img_path)) / 255.0

        factors = self.data[idx]
        tensor_factors = (
            factors
            .replace(
                shape=self.shapes[factors.shape_id % len(self.shapes)],
            )
            .to_tensor(dtype=torch.float32)
        )
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            tensor_factors = self.target_transform(tensor_factors)
        return image, tensor_factors
