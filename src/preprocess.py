"""Dataset preprocessing utilities for sugarcane leaf images."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


def load_image_paths(dataset_dir: Path) -> tuple[list[str], list[int], list[str]]:
    """Scan dataset folder and return image paths, labels, and class names."""
    class_dirs = sorted([path for path in dataset_dir.iterdir() if path.is_dir()])
    class_names = [path.name for path in class_dirs]
    if not class_dirs:
        raise FileNotFoundError(f"No class folders found in dataset directory: {dataset_dir}")

    image_paths: list[str] = []
    labels: list[int] = []

    for label_index, class_dir in enumerate(class_dirs):
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
                image_paths.append(str(image_path))
                labels.append(label_index)

    if not image_paths:
        raise FileNotFoundError(f"No image files found in dataset directory: {dataset_dir}")

    logger.info("Loaded %s images from %s classes", len(image_paths), len(class_names))
    return image_paths, labels, class_names


def split_dataset(image_paths: list[str], labels: list[int], seed: int = 42) -> tuple[list[str], list[int], list[str], list[int], list[str], list[int]]:
    """Split the dataset into training, validation, and test sets."""
    permutation = np.random.RandomState(seed).permutation(len(image_paths))
    shuffled_paths = [image_paths[i] for i in permutation]
    shuffled_labels = [labels[i] for i in permutation]

    total = len(image_paths)
    train_end = int(total * 0.70)
    val_end = int(total * 0.85)

    return (
        shuffled_paths[:train_end],
        shuffled_labels[:train_end],
        shuffled_paths[train_end:val_end],
        shuffled_labels[train_end:val_end],
        shuffled_paths[val_end:],
        shuffled_labels[val_end:],
    )


def decode_image(image_path: tf.Tensor, label: tf.Tensor, img_size: tuple[int, int]) -> tuple[tf.Tensor, tf.Tensor]:
    """Read an image from path, resize, convert to RGB, and normalize."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, img_size)
    label = tf.one_hot(label, depth=5)
    return image, label


def build_dataset(image_paths: list[str], labels: list[int], img_size: tuple[int, int], batch_size: int, shuffle: bool = True) -> tf.data.Dataset:
    """Build a TensorFlow dataset from image paths and labels."""
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths), seed=42)
    dataset = dataset.map(lambda path, label: decode_image(path, label, img_size), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def prepare_datasets(
    dataset_dir: Path,
    img_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    seed: int = 42,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str]]:
    """Prepare training, validation, and test datasets from the dataset directory."""
    image_paths, labels, class_names = load_image_paths(dataset_dir)
    (
        train_paths,
        train_labels,
        val_paths,
        val_labels,
        test_paths,
        test_labels,
    ) = split_dataset(image_paths, labels, seed=seed)

    train_ds = build_dataset(train_paths, train_labels, img_size, batch_size, shuffle=True)
    val_ds = build_dataset(val_paths, val_labels, img_size, batch_size, shuffle=False)
    test_ds = build_dataset(test_paths, test_labels, img_size, batch_size, shuffle=False)

    logger.info("Prepared train/validation/test datasets")
    return train_ds, val_ds, test_ds, class_names
