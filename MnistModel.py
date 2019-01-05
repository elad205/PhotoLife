import mnist
import numpy as np
import torch
import visdom
import VisdomServer
import math
import torchvision.datasets.mnist
import copy
import random
class Layer:
    def __init__(self, layer_tensor, layer_number):
        self.layer = layer_tensor
        self.index = layer_number


class NeuralNetwork(torch.nn.Module):
    def __init__(self, train_data, viz_tool):
        super(NeuralNetwork, self).__init__()
        # using the nvidia cuda api in order to perform calculations on a GPU
        device = torch.device('cuda')
        self.rate = 0.001
        self.loss_plot = []
        self.viz = viz_tool
        # initialize the weights - 4 layers total
        self.weights = []
        self.bias = []

        self.weights.append(torch.randn(2500, 784, device=device, dtype=torch
                                        .float, requires_grad=True))
        self.weights.append(torch.randn(1000, 2500, device=device, dtype=torch
                                        .float, requires_grad=True))
        self.weights.append(torch.randn(500, 1000, device=device, dtype=torch
                                        .float, requires_grad=True))
        self.weights.append(torch.randn(10, 500,  device=device, dtype=torch
                                        .float, requires_grad=True))


        self.init_weights()
        """
        stdv = 1. / math.sqrt(self.weights[0].size(0))
        self.weights[0].data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.weights[1].size(0))
        self.weights[1].data.uniform_(-stdv, stdv)
        """
        for ind in range(150):
            for i, (images, labels) in enumerate(train_data):

                images = images.reshape(-1, 28 * 28).to(device)
                prediction_layer = self.forward_process(Layer(images,0))
                loss = self.softmax_loss(prediction_layer.layer, labels.to(device))
                self.loss_plot.append(i)
                self.loss_plot.append(loss.item())
                self.back_propagation(loss)
            print(loss)
            self.viz.scatter(np.array(self.loss_plot).reshape(int(len(self.loss_plot)/2), 2))

    def forward(self, x):
        out = x.matmul(self.weights[0].t())
        out = out.clamp(min=0)
        out = out.matmul(self.weights[1].t())

        return out

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
        calculated_layer = layer.layer.matmul(self.weights[layer.index].t())
        # perform the activation function on every neuron [relu]
        if layer.index < len(self.weights) - 1:
            activated_layer = calculated_layer.clamp(min=0)
        else:
            activated_layer = calculated_layer
        return Layer(layer_tensor=activated_layer, layer_number=layer.index + 1)

    @staticmethod
    def softmax_loss(prediction_layer, expected_output):
        loss_softmax = torch.nn.CrossEntropyLoss()
        loss = loss_softmax(prediction_layer, expected_output)
        return loss

    def back_propagation(self, loss):
        # perform the backpropagation on the network
        loss.backward()

        with torch.no_grad():
            for index in range(len(self.weights)):
                self.weights[index] -= self.rate * self.weights[index].grad.data

            for index in range(len(self.weights)):
                self.weights[index].grad.zero_()

    @staticmethod
    def convert_label(label):
        output_number = np.array(label)
        output_number = output_number.astype('int64')
        output_number = torch.from_numpy(output_number)
        output_number = output_number.to(torch.device('cuda'))
        return output_number

    def init_weights(self):
        for index in range(len(self.weights)):
            torch.nn.init.kaiming_uniform_(self.weights[index], a=math.sqrt(5))
            self.bias.append(torch.empty(self.weights[index].size(0), requires_grad=True))
        for index in range(len(self.bias)):
            bound = 1 / math.sqrt(self.weights[index].size(1))
            torch.nn.init.uniform_(self.bias[index], -bound, bound)

def main():
    #ser = VisdomServer.Server()
    vis = visdom.Visdom()
    vis.text('Hello, world!')
    train = torchvision.datasets.mnist.MNIST(
        r"..\..\data", train=True, download=True,
        transform=torchvision.transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset=train,
                                               batch_size=100,
                                               shuffle=True)
    test_dataset = torchvision.datasets.MNIST(root='../../data',
                                              train=False,
                                              transform=torchvision.transforms.ToTensor(), download=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=100,
                                              shuffle=False)

    n = NeuralNetwork(train_loader, vis)
    device = torch.device('cuda')

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)
            outputs = n.forward_process(Layer(images, 0))
            _, predicted = torch.max(outputs.layer.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: {} %'.format(
            100 * correct / total))


if __name__ == '__main__':
    main()
