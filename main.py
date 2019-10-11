import tensorflow as tf
import numpy as np
import ipdb
from models.keras_resnet_functional import resnet_v1

BATCH_SIZE = 32
NUM_EPOCHS = 2


def prepare_cifar(x, y, split: str):
    if split == 'train':
        x = tf.image.resize_with_crop_or_pad(x, 40, 40)
        x = tf.image.random_crop(x, [32, 32, 3])
        x = tf.image.random_flip_left_right(x)

    x = tf.cast(x, tf.float32) / 255.
    x = (x - (0.4914, 0.4822, 0.4465)) / (0.2023, 0.1994, 0.2010)
    return x, y


def build_dataset(images: np.ndarray, labels: np.ndarray, split: str):
    ds = tf.data.Dataset.from_tensor_slices((images, labels.squeeze()))

    if split == 'train':
        ds = ds.shuffle(len(labels))

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


def train(model, ds, criterion, optimizer, loss_meter, acc_meter, epoch):
    for i, (images, labels) in enumerate(ds):
        with tf.GradientTape() as tape:
            output = model(images)
            loss = criterion(labels, output)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        loss_meter(loss)
        acc_meter(labels, output)

        print(f'Iter [{i}] loss: {loss_meter.result()} acc: {acc_meter.result() * 100}%')


def test(model, ds, criterion, loss_meter, acc_meter, epoch):
    for i, (images, labels) in enumerate(ds):
        output = model(images)
        loss = criterion(labels, output)

        loss_meter(loss)
        acc_meter(labels, output)

        print(f'Iter [{i}] loss: {loss_meter.result()} acc: {acc_meter.result() * 100}%')


def main():
    train_ds, test_ds = get_datasets()

    model = resnet_v1((32, 32, 3), 20)

    criterion = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=0.2,
        momentum=0.9,
        nesterov=True
    )

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    for epoch in range(NUM_EPOCHS):
        train(model, train_ds, criterion, optimizer, train_loss, train_accuracy, epoch)
        test(model, test_ds, criterion, test_loss, test_accuracy, epoch)

        print(f'Epoch [{epoch}] loss: {test_loss.result()} acc: {test_accuracy.result() * 100}%')

        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        model.save('checkpoint.h5')


if __name__ == '__main__':
    main()
