from collections.abc import Sequence
from typing import Any

from torch.utils.data import DataLoader


def create_loaders(
        cfg,
        datasets,
        **kwargs: Any | Sequence[Any],
):
    """Creates the data loaders."""
    n = len(datasets)
    assert n == 3, "Expected train, val and test dataset."
    # Normalize kwargs: each should be a Sequence of length len(datasets)
    kwargs: dict[str, Sequence] = {
        key: val if isinstance(val, Sequence) else (val,) * n
        for key, val in kwargs.items()
        # Skip iterable DataLoder arguments
        if key not in ("sampler", "batch_sampler")
    }
    assert all(len(val) == n for val in kwargs.values()), "Invalid kwargs"

    train_dataset, val_dataset, test_dataset = datasets
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        **({key: val[0] for key, val in kwargs.items()}),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        **({key: val[1] for key, val in kwargs.items()}),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        **({key: val[2] for key, val in kwargs.items()}),
    )
    return train_loader, val_loader, test_loader
