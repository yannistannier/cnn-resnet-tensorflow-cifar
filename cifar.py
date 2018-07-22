from keras.datasets import cifar10
import cv2
import numpy as np


class Cifar10(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()
        self.train_batch_count = self.x_train.shape[0] // self.batch_size
        self.pp_mean = np.mean(np.concatenate((self.x_train, self.x_test), axis=0), axis=0)
        self.shuffle = np.random.permutation(self.x_train.shape[0])

        self.train_batch_count = self.x_train.shape[0] // self.batch_size
        self.test_batch_count = self.x_test.shape[0] // 125

    def normalize(self, batch_images):
        return (batch_images - self.pp_mean) / 128.0

    def next_test_batch(self, idx):
        batch_images = self.x_test[idx * 125: (idx + 1) * 125]
        batch_labels = self.y_test[idx * 125: (idx + 1) * 125]
        return self.normalize(batch_images), batch_labels

    def next_aug_train_batch(self, idx):
        batch_images = self.x_train[self.shuffle[idx * self.batch_size: (idx + 1) * self.batch_size]]
        batch_labels = self.y_train[self.shuffle[idx * self.batch_size: (idx + 1) * self.batch_size]]

        pad_width = ((0, 0), (4, 4), (4, 4), (0, 0))
        padded_images = np.pad(batch_images, pad_width, mode='constant', constant_values=0)

        aug_batch_images = np.zeros_like(batch_images)

        for i in range(len(batch_images)):
            x = np.random.randint(0, high=8)
            y = np.random.randint(0, high=8)

            cropped_img = padded_images[i][x: x + 32, y: y + 32, :]
            is_flip = np.random.randint(0, high=9)

            if is_flip % 2 == 0:
                flipped_img = cv2.flip(cropped_img, flipCode=1)
            else:
                flipped_img = cropped_img

            np.copyto(aug_batch_images[i], flipped_img)

        return self.normalize(aug_batch_images), batch_labels