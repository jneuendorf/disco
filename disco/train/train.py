"""Training script."""

import inspect
import os
from operator import attrgetter
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from hydra.utils import call, get_object, instantiate
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, Timer
from omegaconf import DictConfig, OmegaConf

from avalanche.benchmarks.scenarios import (
    LazyStreamDefinition,
    create_lazy_generic_benchmark,
)
from avalanche.benchmarks.utils import as_classification_dataset, TransformGroups
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import WandBLogger
from avalanche.training.plugins import EvaluationPlugin
from disco.lightning.callbacks import (
    LoggingCallback,
    MetricsCallback,
    VisualizationCallback,
)
from disco.train.util import train_continually
from idsprites import ContinualBenchmarkRehearsal, ContinualBenchmark
from idsprites.continual_benchmark import BaseContinualBenchmark
from idsprites.infinite_dsprites import Factors, InfiniteDSprites
from idsprites.types import Shape

torch.set_float32_matmul_precision("high")
OmegaConf.register_new_resolver("eval", eval)


# TODO: Infer types from yaml files -> structured configs
#  https://hydra.cc/docs/1.1/tutorials/structured_config/intro/
#  https://pypi.org/project/yaml2pyclass/
#  https://pypi.org/project/yamldataclassconfig/
@hydra.main(config_path="../configs", config_name="main", version_base=None)
def train(cfg: DictConfig) -> None:
    """Train the model in a continual learning setting."""
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    config["job_id"] = os.environ.get("SLURM_JOB_ID")

    shapes = [
        InfiniteDSprites().generate_shape()
        for _ in range(cfg.dataset.tasks * cfg.dataset.shapes_per_task)
    ]
    exemplars = generate_canonical_images(shapes, img_size=cfg.dataset.img_size)
    random_images = generate_random_images(
        shapes,
        img_size=cfg.dataset.img_size,
        factor_resolution=cfg.dataset.factor_resolution,
    )
    callbacks = build_callbacks(cfg, exemplars, random_images)
    trainer = instantiate(cfg.trainer, callbacks=callbacks)
    trainer.logger.log_hyperparams(config)

    strategy = cfg.training.strategy
    if strategy == "naive":
        benchmark = ContinualBenchmark(cfg, shapes=shapes, exemplars=exemplars)
    elif strategy == "rehearsal":
        benchmark = ContinualBenchmarkRehearsal(cfg, shapes=shapes, exemplars=exemplars)
    elif strategy == "rehearsal_only_buffer":
        benchmark = ContinualBenchmarkRehearsal(
            cfg, shapes=shapes, exemplars=exemplars, only_buffer=True
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}.")

    target = get_object(cfg.model._target_)
    if inspect.isclass(target):
        train_ours_continually(cfg, benchmark, trainer)
    elif callable(target):
        train_baseline_continually(cfg, benchmark)
    else:
        raise ValueError(f"Unknown target: {target}.")


def train_ours_continually(cfg, benchmark: BaseContinualBenchmark, trainer):
    """Train our model in a continual learning setting."""
    train_continually(cfg, benchmark, trainer, loader_kwargs=dict(
        drop_last=True,
        shuffle=[True, False, True],  # shuffle 'test' for vis
    ))


def train_baseline_continually(cfg, benchmark: ContinualBenchmark):
    """Train a standard continual learning baseline using Avalanche."""
    model = call(cfg.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.run.define_metric("*", step_metric="Step", step_sync=True)
    get_shape_id = attrgetter('shape_id')
    train_generator = (
        # make_classification_dataset(
        #     dataset=datasets[0],
        #     target_transform=lambda y: y.shape_id,
        # )
        as_classification_dataset(
            dataset=datasets[0],
            transform_groups=TransformGroups.create(
                target_transform=get_shape_id,
            ),
        )
        for datasets, _ in benchmark
    )
    test_generator = (
        as_classification_dataset(
            dataset=datasets[2],
            transform_groups=TransformGroups.create(
                target_transform=get_shape_id,
            ),
        )
        for datasets, _ in benchmark
    )
    train_stream = LazyStreamDefinition(
        train_generator,
        stream_length=benchmark.tasks,
        exps_task_labels=[0] * benchmark.tasks,
    )
    test_stream = LazyStreamDefinition(
        test_generator,
        stream_length=benchmark.tasks,
        exps_task_labels=[0] * benchmark.tasks,
    )
    benchmark = create_lazy_generic_benchmark(train_stream, test_stream)
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    config["job_id"] = os.environ.get("SLURM_JOB_ID")
    loggers = [
        WandBLogger(
            dir=cfg.wandb.save_dir,
            project_name=f"{cfg.wandb.project}_baselines",
            params={"group": cfg.wandb.group},
            config=config,
        ),
    ]

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, experience=True),
        loss_metrics(minibatch=True),
        loggers=loggers,
    )

    if cfg.strategy == "icarl":
        strategy = instantiate(
            cfg.strategy,
            device=device,
            evaluator=eval_plugin,
            optimizer={"params": model.parameters()},
            herding=True,
        )
    else:
        strategy = instantiate(
            cfg.strategy,
            model=model,
            device=device,
            evaluator=eval_plugin,
            optimizer={"params": model.parameters()},
        )
    for train_experience, test_experience in zip(
        benchmark.train_stream, benchmark.test_stream
    ):
        train_task = train_experience.current_experience
        print(f"Task {train_task} train: {len(train_experience.dataset)} samples.")
        print(f"Classes train: {train_experience.classes_in_this_experience}")
        strategy.train(train_experience)

        test_task = test_experience.current_experience
        if (
            not cfg.training.test_once
            and test_task % cfg.training.test_every_n_tasks == 0
        ):
            print(f"Task {test_task} test: {len(test_experience.dataset)} samples.")
            min_class_id = min(test_experience.classes_in_this_experience)
            max_class_id = max(test_experience.classes_in_this_experience)
            print(f"Classes test: {min_class_id}-{max_class_id}")
            strategy.eval(test_experience)


def generate_canonical_images(shapes, img_size: int) -> list[Shape]:
    """Generate a batch of exemplars for training and visualization."""
    dataset = InfiniteDSprites(
        img_size=img_size,
    )
    return [
        dataset.draw(
            Factors(
                color=(1.0, 1.0, 1.0),
                shape=shape,
                shape_id=None,
                scale=1.0,
                orientation=0.0,
                position_x=0.5,
                position_y=0.5,
            )
        )
        for shape in shapes
    ]


def generate_random_images(
    shapes: list, img_size: int, factor_resolution: int, num_imgs: int = 25
):
    """Generate a batch of images for visualization."""
    scale_range = np.linspace(0.5, 1.0, factor_resolution)
    orientation_range = np.linspace(0, 2 * np.pi, factor_resolution)
    position_x_range = np.linspace(0, 1, factor_resolution)
    position_y_range = np.linspace(0, 1, factor_resolution)
    dataset = InfiniteDSprites(
        img_size=img_size,
        shapes=shapes,
        scale_range=scale_range,
        orientation_range=orientation_range,
        position_x_range=position_x_range,
        position_y_range=position_y_range,
    )
    return [dataset.draw(dataset.sample_factors()) for _ in range(num_imgs)]


def build_callbacks(cfg: DictConfig, canonical_images: list, random_images: list):
    """Prepare the appropriate callbacks."""
    callbacks = []
    callback_names = cfg.training.callbacks
    if "checkpointing" in callback_names:
        callbacks.append(
            ModelCheckpoint(
                dirpath=Path(cfg.trainer.default_root_dir)
                / os.environ.get("SLURM_JOB_ID"),
                every_n_epochs=cfg.training.checkpoint_every_n_tasks
                * cfg.training.epochs_per_task,
                save_top_k=-1,
                save_weights_only=True,
            )
        )
    if "learning_rate_monitor" in callback_names:
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    if "logging" in callback_names:
        callbacks.append(LoggingCallback())
    if "metrics" in callback_names:
        callbacks.append(
            MetricsCallback(
                log_train_accuracy=cfg.training.log_train_accuracy,
                log_val_accuracy=cfg.training.log_val_accuracy,
                log_test_accuracy=cfg.training.log_test_accuracy,
            )
        )
    if "timer" in callback_names:
        callbacks.append(Timer())
    if "visualization" in callback_names:
        callbacks.append(VisualizationCallback(canonical_images, random_images))
    return callbacks


if __name__ == "__main__":
    train()
