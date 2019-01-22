import skimage
import torch
import visdom
import numpy
import torchvision


class DataParser:
    def __init__(self, batch_size, width_height):
        self.feature_list = []
        self.label_list = []
        self.batch_size = batch_size
        self.image_axe = width_height

    def load_cifar10(self):
        """
        this function loads the cifar10 database which contains various images
        into the script.
        :return a data loader object, a tuple for the training data and the test data.
        """
        train_dataset = torchvision.datasets.CIFAR10(
            "data", train=True, download=True,
            transform=torchvision.transforms.ToTensor())

        # load training data
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=True)

        test_dataset = torchvision.datasets.CIFAR10(root='data',
                                                  train=False,
                                                  transform=torchvision.transforms.
                                                  ToTensor())
        # load test data
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=self.batch_size,
                                                  shuffle=False)

        return train_loader, test_loader

    def parse_data(self, data_loader):
        """
        this function parses the data and splits it into features and labels.
        It does that by converting the image into a numpy array and than performing operations
        on the array in order to turn it into the structure required for skimage to convert the pic.
        after that it returns the arrays to torch tensors and adds them to a list
        :param data_loader: a dataloader object containing the data
        :return:
        """
        for i, (images, labels) in enumerate(data_loader):
            # convert the images into numpy arrays
            images = images.numpy()
            
            # swap the axes because skimage uses x , x , c format and torch uses c, x, x format
            images = numpy.swapaxes(images, 1, 3)

            # perform the casting from rgb to lab and convert the image to values lower than 1.
            lab_image = skimage.color.rgb2lab(1.0 / 255 * images)

            # extract the features
            features = lab_image[:, :, :, 0].reshape(self.batch_size, self.image_axe, self.image_axe, 1)

            # extract the labels
            labels = lab_image[:, :, :, 1:]

            # make the label values smaller than 1
            labels = labels / 128

            # convert the data back to torch format
            features = torch.FloatTensor(numpy.swapaxes(features, 1, 3).tolist())
            labels = torch.FloatTensor(numpy.swapaxes(labels, 1, 3).tolist())

            # append the data to the objects
            self.feature_list.append(features)
            self.label_list.append(labels)


def main():
    a = DataParser(30, 32)
    im = a.load_cifar10()
    a.parse_data(im[0])


if __name__ == '__main__':
    main()
