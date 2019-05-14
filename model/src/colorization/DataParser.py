import torch
import numpy
import torchvision
import cv2
from torchvision import transforms
import os
import sys


class PlacesDataSet(torchvision.datasets.ImageFolder):
    def __init__(self, train_dir, transf, norm=True):
        super(PlacesDataSet, self).__init__(train_dir, transform=transf)
        self.norm = norm

    def __getitem__(self, index):
        path = self.samples[index][0]
        sample = self.loader(path)
        augment = transforms.Compose([
            transforms.RandomHorizontalFlip()])
        normilize = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(
                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        sample = augment(sample)

        if self.norm:
            sample = normilize(sample)
            sample = sample.numpy()
            sample = sample.transpose((1, 2, 0))
        else:
            sample = numpy.array(sample)

        feature, label = DataParser.rgb_parse(sample)
        feature = self.transform(feature)
        label = self.transform(label)
        return feature, label


class DataParser:
    def __init__(self):
        self.feature_list = []
        self.label_list = []

    def __next__(self):

        # get to the next item
        self.batch_counter += 1

        # stop when iterated over all of the objects
        if self.batch_counter == len(self.feature_list):
            raise StopIteration

        # return the images and labels
        return self.feature_list[self.batch_counter], self.label_list[
            self.batch_counter]

    def __iter__(self):
        # create a counter to index the lists
        self.batch_counter = -1
        return self

    @staticmethod
    def load_places_test(batch_size, test_dir):
        if not os.path.exists(test_dir):
            print("cant find test dataset", file=sys.stderr)
            exit(-1)

        test_dataset = PlacesDataSet(
            test_dir,
            transforms.Compose([
                transforms.ToTensor()]), norm=False)

        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=1)

        return test_loader

    @staticmethod
    def load_places_train(batch_size, train_dir):

        if not os.path.exists(train_dir):
            print("cant find train dataset", file=sys.stderr)
            exit(-1)

        train_dataset = PlacesDataSet(
            train_dir, transforms.Compose([transforms.ToTensor()]))

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=2, pin_memory=True)

        return train_loader

    @staticmethod
    def rgb_parse(image):
        rgb_image = (image / 255.0).astype('float32')
        gray_image = cv2.cvtColor(image.astype('float32'),
                                  cv2.COLOR_RGB2GRAY) / 255.0
        return gray_image, rgb_image

    @staticmethod
    def load_images(paths: list):
        arr = []
        for path in paths:
            try:
                arr.append(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
            except cv2.error:
                print(
                    "an error accrued while opening the images,"
                    " please check the validity of the images", file=sys.stderr)
                exit(-1)

            arr[-1] = cv2.resize(arr[-1], (256, 256)).reshape((1, 256, 256))

        arr = numpy.asarray(arr).astype('float32') / 255.0
        return arr



