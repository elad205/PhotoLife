import torch
import numpy
import torchvision
import cv2
from torchvision import transforms
import progressbar


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
    def load_cifar10(batch_size):
        """
        this function loads the cifar10 database which contains various images
        into the script.
        :return a data loader object, a tuple for the training data and the
        test data.
        """
        train_dataset = torchvision.datasets.CIFAR10(
            root="cifar10", train=True,
            download=True, transform=torchvision.transforms.ToTensor())

        # load training data
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True)

        test_dataset = torchvision.datasets.CIFAR10(
            root='cifar10',
            train=False, transform=torchvision.transforms.ToTensor())
        # load test data
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

    @staticmethod
    def load_places_dataset(batch_size):
        train_dir = r'D:\data_256'
        test_dir = r'D:\data_test_256'

        train_dataset = torchvision.datasets.ImageFolder(
            train_dir,
            transforms.Compose([transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor()]))
        test_dataset = torchvision.datasets.ImageFolder(
            test_dir,
            transforms.Compose([transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor()]))

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

    def parse_data(self, data_loader, stopper=None):
        """
        this function parses the data and splits it into features and labels.
        It does that by converting the image into a numpy array and than
        performing operations on the array in order to turn it into the
        structure required for skimage to convert the pic.
        after that it returns the arrays to torch tensors and adds them to a
        list.
        :param data_loader: a data loader object containing the data
        :param stopper: a number that indicates when to stop loading data
        to the memory
        :return:
        """
        prog = progressbar.ProgressBar(
            widgets=['parsing data ', progressbar.SimpleProgress(), ' ',
                     progressbar.ETA()],
            max_value=len(
                data_loader) * 16 if stopper is None else stopper * 16)

        for i, (images, labels) in enumerate(data_loader):
            # used to get only a fraction of the dataset
            if stopper is not None and i > stopper:
                break
            # convert the images into numpy arrays
            images = images.numpy()
            # swap the axes because cv2 uses x , x , c format and torch
            # uses c, x, x format
            images = images.transpose((0, 2, 3, 1))
            # perform the casting from rgb to lab and convert the image to
            # values lower than 1.
            lab_image = []
            for image in images:
                lab_image.append(cv2.cvtColor(image, cv2.COLOR_RGB2LAB))

            lab_image = numpy.asarray(lab_image)
            # extract the features
            a_dim = lab_image.shape[0]
            b_dim = lab_image.shape[1]
            c_dim = lab_image.shape[2]
            features = lab_image[:, :, :, 0].reshape(a_dim, b_dim, c_dim, 1)
            # convert the structure back to c,x,x
            features = features.transpose((0, 3, 1, 2))
            # extract the labels
            labels = lab_image[:, :, :, 1:]

            # make the label values smaller than 1
            labels = labels / 128
            # convert the structure back to c,x,x
            labels = labels.transpose((0, 3, 1, 2))
            # convert the data back to torch format
            features = torch.from_numpy(features)
            labels = torch.from_numpy(labels)
            prog.update(i * a_dim)
            # append the data to the objects
            self.feature_list.append(features)
            self.label_list.append(labels)
            del lab_image

        prog.finish()

    def rgb_parse(self, data_loader):
        for images in data_loader:
            images = images.numpy()
            self.feature_list.append(cv2.cvtColor(images, cv2.COLOR_RGB2GRAY))
            self.label_list.append(images)

    @staticmethod
    def reconstruct_image(feature, result):
        result = result * 128
        f = feature.cpu().numpy()
        r = result.cpu().numpy()
        canvas = numpy.zeros((f.shape[0], 3, 224, 224), dtype='float32')
        canvas[:, 0, :, :] = f.reshape(f.shape[0], 224, 224)
        canvas[:, 1:, :, :] = r
        lst = []
        canvas = canvas.transpose((0, 2, 3, 1))
        for image in canvas:
            lst.append(cv2.cvtColor(image, cv2.COLOR_LAB2RGB))

        return numpy.asarray(lst).transpose((0, 3, 1, 2))


def main():
    ob = DataParser()
    ob.parse_data()


if __name__ == '__main__':
    main()
