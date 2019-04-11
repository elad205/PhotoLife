import torch
import numpy
import torchvision
import cv2
from torchvision import transforms
import progressbar


class PlacesDataSet(torchvision.datasets.ImageFolder):
    def __init__(self, train_dir, transf):
        super(PlacesDataSet, self).__init__(train_dir, transform=transf)

    def __getitem__(self, index):
        path = self.samples[index][0]
        sample = self.loader(path)
        augment = transforms.Compose([transforms.RandomHorizontalFlip()])
        sample = augment(sample)
        sample = numpy.array(sample)
        feature, label = DataParser.cvt_numpy_image(sample)
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
            dataset=train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=8, pin_memory=True)

        test_dataset = torchvision.datasets.CIFAR10(
            root='cifar10',
            train=False, transform=torchvision.transforms.ToTensor())
        # load test data
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

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
            shuffle=True, num_workers=4, pin_memory=True)

        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=4)

        return train_loader, test_loader

    @staticmethod
    def create_loading_bar(data_loader, stopper):
        prog = progressbar.ProgressBar(
            widgets=['parsing data ', progressbar.SimpleProgress(), ' ',
                     progressbar.ETA()],
            max_value=len(
                data_loader) * 4 if stopper is None else stopper * 4)
        return prog

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
    def reconstruct_image(feature, result):
        result = result * 128
        f = feature.cpu().clone().detach().numpy()
        r = result.cpu().clone().detach().numpy()
        canvas = numpy.zeros((f.shape[0], 3, 224, 224), dtype='float32')
        canvas[:, 0, :, :] = f.reshape(f.shape[0], 224, 224)
        canvas[:, 1:, :, :] = r
        lst = []
        canvas = canvas.transpose((0, 2, 3, 1))
        for image in canvas:
            lst.append(cv2.cvtColor(image, cv2.COLOR_LAB2RGB))

        return numpy.asarray(lst).transpose((0, 3, 1, 2))

    @staticmethod
    def torch_rgb_to_lab(image_batch, device):

        # convert the images to a large rgb image
        all_pixels = image_batch.view((-1, 3)).to(device)
        all_pixels = all_pixels.type(torch.float32)

        # create a mask for pixels which values are smaller than 0.4045
        linear_mask = all_pixels <= 0.04045
        linear_mask = linear_mask.type(torch.float32)

        # create a mask for pixels which values are larger than 0.4045
        exp_mask = all_pixels > 0.04045
        exp_mask = exp_mask.type(torch.float32)

        pixels = (all_pixels / 12.92 * linear_mask) + (
                ((all_pixels + 0.55) / 1.055) ** 2.4) * exp_mask

        conv_xyz = torch.Tensor([
            #    X        Y          Z
            [0.412453, 0.212671, 0.019334],  # R
            [0.357580, 0.715160, 0.119193],  # G
            [0.180423, 0.072169, 0.950227],  # B
        ]).to(device)

        # perform the transformation
        xyz_pixels = torch.matmul(pixels, conv_xyz)

        # normalize the pixels to d65
        xyz_pixels[:, 0] *= 1 / 0.950456
        xyz_pixels[:, 2] *= 1 / 1.088754

        epsilon = 6 / 29

        # create a linear mask for the pixels
        linear_mask = xyz_pixels <= epsilon ** 3
        linear_mask = linear_mask.type(torch.float32)

        # create an exponential mask for the pixels
        exp_mask = xyz_pixels > epsilon ** 3
        exp_mask = exp_mask.type(torch.float32)

        # calculate the values according to the formula
        fxfyfz_pixels = (xyz_pixels / (
                    3 * epsilon ** 2) + 4 / 29) * linear_mask + (
                                xyz_pixels ** (1 / 3)) * exp_mask

        conv_lab = torch.Tensor([
            #  l       a       b
            [0.0, 500.0, 0.0],       # fx
            [116.0, -500.0, 200.0],  # fy
            [0.0, 0.0, -200.0],      # fz
        ]).to(device)

        # perform the transformation
        lab_pixels = torch.matmul(fxfyfz_pixels, conv_lab) + torch.Tensor(
            [-16.0, 0.0, 0.0]).to(device)

        final_images = lab_pixels.view(image_batch.size())

        final_images[:, :, :, 0] /= 50
        final_images[:, :, :, 0] -= 1
        final_images[:, :, :, 1:] /= 121

        return DataParser.torch_lab_to_gray(final_images, device), final_images

    @staticmethod
    def torch_lab_to_gray(image_batch, device):
        image_batch_gray = image_batch[:, 0, :, :].unsqueeze(1)
        return image_batch_gray.to(device)

def main():
    ob = DataParser()
    ob.parse_data()


if __name__ == '__main__':
    main()
