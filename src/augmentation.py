"""Data augmentation utilities for TensorFlow image training."""

from __future__ import annotations

import tensorflow as tf


def augmentation_pipeline(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Apply data augmentation to a single image."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    image = tf.image.random_saturation(image, lower=0.75, upper=1.25)
    image = tf.image.random_hue(image, max_delta=0.05)
    image = tf.clip_by_value(image, 0.0, 1.0)
    image = tf.image.convert_image_dtype(image, tf.uint8)
    image = tf.image.random_jpeg_quality(image, 80, 100)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def apply_augmentation(dataset: tf.data.Dataset, batch_size: int = 32) -> tf.data.Dataset:
    """Add augmentation to the training dataset.

    This function expects a batched dataset and will unbatch it before
    applying image-level augmentation, then batch it again for training.
    """
    augmented_dataset = dataset.unbatch()
    augmented_dataset = augmented_dataset.map(augmentation_pipeline, num_parallel_calls=tf.data.AUTOTUNE)
    augmented_dataset = augmented_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return augmented_dataset
