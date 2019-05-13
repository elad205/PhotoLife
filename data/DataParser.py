import torch
import numpy
import torchvision
import cv2
from torchvision import transforms


class PlacesDataSet(torchvision.datasets.ImageFolder):
    def __init__(self, train_dir, transf):
        super(PlacesDataSet, self).__init__(train_dir, transform=transf)

    def __getitem__(self, index):
        path = self.samples[index][0]
        sample = self.loader(path)
        augment = transforms.Compose([transforms.RandomHorizontalFlip()])
        sample = augment(sample)
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
    def load_places_dataset(batch_size):
        train_dir = r'C:\data_256'
        test_dir = r'C:\data_test_256'

        train_dataset = PlacesDataSet(
            train_dir,
            transforms.Compose([
                                transforms.ToTensor()]))
        test_dataset = torchvision.datasets.ImageFolder(
            test_dir,
            transforms.Compose([transforms.ToTensor()]))

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=2, pin_memory=True)

        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=1)

        return train_loader, test_loader

    @staticmethod
    def convert_to_lab(image_batch, opposite=True):
        image_batch = image_batch.numpy()
        image_batch = image_batch.transpose((0, 2, 3, 1))
        lab_image = numpy.empty_like(image_batch)
        gray_images = numpy.empty_like(image_batch[..., :1])
        if not opposite:
            image_batch[:, :, :, 0] += 1
            image_batch[:, :, :, 0] *= 50
            image_batch[:, :, :, 1:] *= 127

        for index in range(image_batch.shape[0]):
            if opposite:
                lab_image[index, :, :, :] = \
                    cv2.cvtColor(image_batch[index], cv2.COLOR_RGB2Lab)
                gray_images[index, :, :, 0] = \
                    cv2.cvtColor(image_batch[index], cv2.COLOR_RGB2GRAY)
            else:
                lab_image[index, :, :, :] = cv2.cvtColor(
                    image_batch[index], cv2.COLOR_Lab2RGB)

        if not opposite:
            return lab_image.transpose((0, 3, 1, 2))

        a_dim = gray_images.shape[0]
        b_dim = gray_images.shape[1]
        c_dim = gray_images.shape[2]

        gray_images = gray_images.reshape((a_dim, b_dim, c_dim, 1))
        if opposite:
            lab_image[:, :, :, 0] /= 50
            lab_image[:, :, :, 0] -= 1
            lab_image[:, :, :, 1:] /= 127

        return gray_images.transpose((0, 3, 1, 2)), lab_image.transpose(
            (0, 3, 1, 2))

    @staticmethod
    def cvt_numpy_image(image):
        image = image / 255.0
        image = image.astype('float32')
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
        gray_image = lab_image[:, :, 0]
        gray_image = gray_image / 100
        lab_image[:, :, 0] /= 50
        lab_image[:, :, 0] -= 1
        lab_image[:, :, 1:] /= 127
        return gray_image, lab_image

    @staticmethod
    def reconstruct_image(l_tensor, ab_tensor):
        canvas = torch.zeros((l_tensor.size(0), 3, 256, 256))
        canvas[:, 0, :, :] = torch.squeeze(l_tensor)
        canvas[:, 0, :, :] *= 2
        canvas[:, 0, :, :] -= 1
        canvas[:, 1:, :, :] = ab_tensor

        return canvas

    @staticmethod
    def rgb_parse(image):
        rgb_image = (image / 255.0).astype('float32')
        gray_image = cv2.cvtColor(image.astype('float32'),
                                  cv2.COLOR_RGB2GRAY) / 255.0
        """
        gray_image_3 = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)

        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YUV)

        gray_image_3 = cv2.cvtColor(gray_image_3, cv2.COLOR_RGB2YUV)
        gray_image_3[:, :, 1:3] = rgb_image[:, :, 1:3]

        """
        return gray_image, rgb_image


def main():
    pass


if __name__ == '__main__':
    main()
