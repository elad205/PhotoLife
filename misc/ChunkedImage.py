import numpy
from DataParser import DataParser


class InvalidFormatException(Exception):
    def __init__(self):
        super(InvalidFormatException, self).__init__(
            "image format should be a square and divisible by chunk size")


class ChunkImage(object):
    def __init__(self, im, chunk_size=2):
        self.image = im
        self.chunk_size = chunk_size
        if len(self.image.shape) != 4 \
                and self.image.shape[1] != self.image.shape[2] \
                and self.image.shape[2] % self.chunk_size != 0:
            raise InvalidFormatException
        self.image_size = self.image.shape[2]

    def __iter__(self):

        self.counter = [0, 0, 0]
        return self

    def __len__(self):

        return self.image.shape

    def __next__(self):

        if self.counter[0] == self.image.shape[0]:
            raise StopIteration

        ret_value = self.image[self.counter[0]][
                    self.counter[2]: self.counter[2] + self.chunk_size,
                    self.counter[1]: self.counter[1] + self.chunk_size, 0:]

        self.counter[1] += self.chunk_size

        if self.counter[1] == self.image_size:
            self.counter[2] += self.chunk_size
            self.counter[1] = 0

        if self.counter[2] == self.image_size:
            self.counter[2] = 0
            self.counter[1] = 0
            self.counter[0] += 1

        return ret_value

    def __getitem__(self, item):

        return self.image[item]

    @staticmethod
    def generate_objects(data):
        imr = []
        for i, (image, lables) in enumerate(data):
            image = numpy.transpose(image.numpy(), (0, 2, 3, 1))
            imr.append(ChunkImage(image))

        return imr


if __name__ == '__main__':
    data = DataParser.load_cifar10(200)
    imr = []
    for i, (image, lables) in enumerate(data[0]):
        imr.append(ChunkImage(image))

    for items in imr:
        for chuncks in items:
            pass
            #print(chuncks.shape)
