import mnist
import numpy as np
import torch
import visdom
import VisdomServer
import math
import torchvision


class Layer:
    def __init__(self, layer_tensor, layer_number):
        self.layer = layer_tensor
        self.index = layer_number


class NeuralNetwork:
    def __init__(self, input_layer, output_label):

        # using the nvidia cuda api in order to perform calculations on a GPU
        device = torch.device('cuda')
        self.cost = (0, 0)
        self.rate = 1e-6

        # initialize the weights - 4 layers total
        self.weights = []
        self.weights.append(torch.randn(784, 2500, device=device, dtype=torch
                                        .double, requires_grad=True))
        self.weights.append(torch.randn(2500, 1000, device=device, dtype=torch
                                        .double, requires_grad=True))
        self.weights.append(torch.randn(1000, 500, device=device, dtype=torch
                                        .double, requires_grad=True))
        self.weights.append(torch.randn(500, 10,  device=device, dtype=torch
                                        .double, requires_grad=True))

        for index in range(0, len(input_layer), 100):
            feature = input_layer[index: index + 100]
            output_number = output_label[index: index + 100]
            prediction_layer = self.forward_process(
                Layer(self.parse_python_input_to_cuda_tensor(feature, 100, 784), 0))
            loss = self.softmax_loss(prediction_layer.layer,
                                     self.convert_label(output_number))
            print(loss)
            self.back_propagation(loss)

    def forward_process(self, layer):
        """
        this is the forward iteration of the network
        :param layer:
        :return:
        """
        if layer.index == len(self.weights):
            return layer
        else:
            new_layer = self.create_activated_layer(layer)
            layer_1 = self.forward_process(new_layer)
            return layer_1

    def create_activated_layer(self, layer):
        """
        creates activated layer using the relu function
        :param layer: layer objects
        :return: activated layer
        """
        # create the layer of the model
        calculated_layer = layer.layer.mm(self.weights[layer.index])
        # perform the activation function on every neuron [relu]
        activated_layer = calculated_layer.clamp(min=0)
        return Layer(activated_layer, layer.index + 1)

    @staticmethod
    def parse_python_input_to_cuda_tensor(lst, batch_size, input_size):
        """
        this function gets a python list and parses it into a pytorch tensor
        :param lst:  the python list
        :param batch_size: the batch size of the input
        :param input_size: the size of each input
        :return: parsed pytorch tensor
        """
        temp = np.array(lst).reshape(batch_size, input_size)
        temp = temp.astype('float')
        torch_tensor = torch.from_numpy(temp)
        torch_tensor = torch_tensor.to(torch.device('cuda'))
        return torch_tensor

    @staticmethod
    def softmax_loss(prediction_layer, expected_output):
        loss_softmax = torch.nn.CrossEntropyLoss()
        # currently does loss calculation in the cpu in order to
        # avoid gradient explosion
        prediction_layer = prediction_layer.to(torch.device('cpu'))
        loss = loss_softmax(prediction_layer, expected_output)
        return loss

    def back_propagation(self, loss):
        # perform the backpropagation on the network
        torch.autograd.backward(loss)

        with torch.no_grad():
            for index in range(len(self.weights)):
                self.weights[index] -= self.rate * self.weights[index].grad
                self.weights[index].grad.zero_()

    @staticmethod
    def convert_label(label):
        output_number = np.array(label)
        output_number = output_number.astype('int64')
        output_number = torch.from_numpy(output_number)
        output_number = output_number.to(torch.device('cpu'))
        return output_number


def main():
    ser = VisdomServer.Server()
    vis = visdom.Visdom()
    vis.text('Hello, world!')
    data = mnist.MNIST(r"C:\Users\eladc\PycharmProjects\PyTorchEnv\MNIST")
    images, labels = data.load_training()
    print(labels[0])
    # print(np.asarray(images)[0].reshape(28,28))
    img = np.array(images[0]).astype('uint8').reshape(1, 28, 28)
    vis.image(img)
    img = np.array(images[0]).reshape(-1, 784)
    img = img.astype('float')
    print(img.dtype)
    n = NeuralNetwork(images, labels)


if __name__ == '__main__':
    main()
