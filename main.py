import tensorflow as tf
import numpy as np
import ipdb

BATCH_SIZE = 32


def prepare_cifar(x, y, split: str):
    if split == 'train':
        x = tf.image.resize_with_crop_or_pad(x, 40, 40)
        x = tf.image.random_crop(x, [32, 32, 3])
        x = tf.image.random_flip_left_right(x)

    x = tf.cast(x, tf.float32) / 255.
    x = (x - (0.4914, 0.4822, 0.4465)) / (0.2023, 0.1994, 0.2010)
    return x, y[0]


def build_dataset(images: np.ndarray, labels: np.ndarray, split: str):
    ds = tf.data.Dataset.from_tensor_slices((images, labels))

    ds = ds.map(lambda x, y: prepare_cifar(x, y, split), num_parallel_calls=-1)

    ds = ds.prefetch(-1)
    ds = ds.batch(BATCH_SIZE)

    return ds


def get_datasets(name='cifar10'):
    cifar = getattr(tf.keras.datasets, name)
    (x_train, y_train), (x_test, y_test) = cifar.load_data()
    train_ds = build_dataset(x_train, y_train, 'train')
    test_ds = build_dataset(x_test, y_test, 'test')
    return train_ds, test_ds


def main():
    train_ds, test_ds = get_datasets()

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(10, 3, padding='same'),
        tf.keras.layers.GlobalAveragePooling2D()
    ])

    criterion = tf.keras.losses.SparseCategoricalCrossentropy()

    for images, labels in train_ds:
        y = model(images)
        print(y.shape)
        print(labels)
        loss = criterion(labels, y)
        print(loss)
        ipdb.set_trace()
        break


if __name__ == '__main__':
    main()
