from collections.abc import Sequence
from pathlib import Path
from typing import Any, Union, TypeVar

import numpy as np
from hydra.utils import instantiate
from lightning import Trainer
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image

from disco.lightning.modules import ContinualModule
from idsprites.continual_benchmark import BaseContinualBenchmark
from idsprites.types import Floats

TRAIN = TypeVar('TRAIN', bound=Dataset)
VAL = TypeVar('VAL', bound=Dataset)
TEST = TypeVar('TEST', bound=Dataset)


def train_continually(
    cfg,
    benchmark: BaseContinualBenchmark[TRAIN, VAL, TEST],
    trainer: Trainer,
    loader_kwargs: dict[str, Any],
):
    """Train the model in a continual learning setting."""
    test_loader: DataLoader | None = None
    model: ContinualModule = instantiate(cfg.model)
    for task_id, (datasets, task_exemplars) in enumerate(benchmark):
        if cfg.training.reset_model:
            model = instantiate(cfg.model)
        model.task_id = task_id
        train_loader, val_loader, test_loader = create_loaders(
            cfg,
            datasets,
            **loader_kwargs,
        )

        for exemplar in task_exemplars:
            model.add_exemplar(exemplar)

        if cfg.training.validate:
            trainer.fit(model, train_loader, val_loader)
        else:
            trainer.fit(model, train_loader)
        if (
            not cfg.training.test_once
            and task_id % int(cfg.training.test_every_n_tasks) == 0
        ):
            trainer.test(model, test_loader)
        trainer.fit_loop.max_epochs += cfg.trainer.max_epochs

    if test_loader:
        trainer.test(model, test_loader)


def create_loaders(
    cfg,
    datasets: tuple[TRAIN, VAL, TEST],
    **kwargs,
) -> tuple[DataLoader[TRAIN], DataLoader[VAL], DataLoader[TEST]]:
    """Creates the data loaders."""
    n = len(datasets)
    assert n == 3, "Expected train, val and test dataset."
    # Normalize kwargs: each should be a Sequence of length len(datasets)
    seq_kwargs: dict[str, Sequence] = {
        key: val if isinstance(val, Sequence) else (val,) * n
        for key, val in kwargs.items()
        # Skip iterable DataLoder arguments
        if key not in ("sampler", "batch_sampler")
    }
    assert all(len(val) == n for val in seq_kwargs.values()), "Invalid kwargs"

    train_dataset, val_dataset, test_dataset = datasets
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        **({key: val[0] for key, val in seq_kwargs.items()}),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        **({key: val[1] for key, val in seq_kwargs.items()}),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        **({key: val[2] for key, val in seq_kwargs.items()}),
    )
    return train_loader, val_loader, test_loader


def read_img_to_np(path: Union[Path, str]) -> Floats:
    """Read an image and normalize it to [0, 1].
    Args:
        path: The path to the image.
    Returns:
        The image as a numpy array.
    """
    return np.array(read_image(str(path)) / 255.0)
