# Copyright 2018 Du Fengtong

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from tensorflow.python.framework import dtypes
import h5py
import matplotlib.pyplot as plt

def add_image_noise(sigma,image):
    '''add gaussian noise to input image, or images with in an array'''
    mean = 0
    gauss = np.random.normal(mean, sigma, image.shape)
    gauss = gauss.reshape(image.shape)
    noisy_im = image + gauss
    cliped_noisy_im = np.clip(noisy_im, 0, 255)
    return cliped_noisy_im

def read_h5_data(h5_path, reshape=False):
    '''
    an alternative version of tensorflow.examples.tutorials.mnist.inputdata.read_data_sets()
    h5_path: path of the noisy mnist dataset h5 file
    return tensorflow Dataset instance, can be used with
    batch_X, batch_Y = noisy_mnist.train.next_batch(100)
    '''

    datasets = h5py.File(h5_path, "r")
    train_images = datasets['train_images'][:,:]
    test_images = datasets['test_images'][:,:]
    validation_images = datasets['val_images'][:,:]
    train_labels = datasets['train_labels'][:,:]
    test_labels = datasets['test_labels'][:,:]
    validation_labels = datasets['val_labels'][:,:]

    train = DataSet(
        train_images, train_labels, dtype=dtypes.float32, reshape=reshape, seed=None)
    validation = DataSet(
        validation_images,
        validation_labels,
        dtype=dtypes.float32,
        reshape=reshape,
        seed=None)
    test = DataSet(
        test_images, test_labels, dtype=dtypes.float32, reshape=reshape, seed=None)

    return base.Datasets(train=train, validation=validation, test=test)

def save_h5_data(file_name, train_x, train_y, val_x, val_y, test_x, test_y):
    f = h5py.File(file_name, 'w')
    f.create_dataset('train_images', data=train_x)
    f.create_dataset('train_labels', data=train_y)
    f.create_dataset('val_images', data=val_x)
    f.create_dataset('val_labels', data=val_y)
    f.create_dataset('test_images', data=test_x)
    f.create_dataset('test_labels', data=test_y)

def show_image_sample(sigma, images, labels):
    labels = np.argmax(labels, axis=1)
    unique_labels = np.unique(labels)
    label_list = unique_labels.tolist()
    str_label_list = [str(x) for x in label_list]
    image_dict = {}
    fig, ax = plt.subplots(nrows=2, ncols=5)
    for i in range(images.shape[0]):
        if len(label_list) == 0:
            break
        for j, label in enumerate(label_list):
            if labels[i] == label:
                image_dict[str(label)] = images[i]
                del label_list[j]
    row, col, c = images[0].shape
    for label, image in image_dict.items():
        idx = str_label_list.index(label)
        image = image.reshape(row, col)
        ax_r = int(idx/5)
        ax_c = idx - ax_r*5
        ax[ax_r][ax_c].imshow(image, cmap='gray')
        ax[ax_r][ax_c].set_axis_off()
    im_name = 'data/samples/sigma_%.2f.png' % sigma
    plt.show()
    fig.savefig(im_name, dpi=fig.dpi)

def create_noisy_mnist(sigma, save_dir):
    mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

    train_images = np.multiply(mnist.train.images, 255.0)
    val_images = np.multiply(mnist.validation.images, 255.0)
    test_images = np.multiply(mnist.test.images, 255.0)
    train_labels = mnist.train.labels
    val_labels = mnist.validation.labels
    test_labels = mnist.test.labels

    noisy_test_images = add_image_noise(sigma, test_images)

    show_image_sample(sigma, noisy_test_images, test_labels)
    f_path = '%s/noisy_mnist_sigma_%d.hdf5' % (save_dir, sigma)
    save_h5_data(f_path, train_images, train_labels, val_images, val_labels, noisy_test_images, test_labels)

if __name__ == '__main__':
    dir = 'D:\dft\program\predropout-tensorflow\data'
    create_noisy_mnist(64, dir)
