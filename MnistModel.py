import numpy as np
import torch
import visdom
import math
import torchvision.datasets.mnist
import copy


class Layer:
    """
    this class represents a layer in the network, it contains the
    layer tensor and the place of the layer in the network
    """
    def __init__(self, layer_tensor, layer_number):
        self.layer = layer_tensor
        self.index = layer_number


class NeuralNetwork(torch.nn.Module):
    def __init__(self, viz_tool, layer_number):
        """
        initialize the network variables
        :param viz_tool: visdom object to display plots
        :param layer_number: the number of layers in the network
        """
        super(NeuralNetwork, self).__init__()
        # using the nvidia cuda api in order to perform calculations on a GPU
        with torch.no_grad():
            self.device = torch.device('cuda')
            self.rate = 0.001
            self.loss_plot = []
            self.cost_plot = []
            self.viz = viz_tool
            self.network_layers = layer_number
            self.weights = torch.nn.ModuleList()
        # initiate the weights
        self.init_weights_linear_profile([(784, 2500), (2500, 1000),
                                          (1000, 500), (500, 10)])

        # create an optimizer for the network
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.rate)

    def train_model(self, epoches, train_data):
        """
        train the model
        :param epoches: the number of epoches to perform
        :param train_data: the dataset
        :return:
        """
        loss = None
        for epoch in range(epoches):
            # run on all the data examples parsed by pytorch vision
            for i, (images, labels) in enumerate(train_data):
                # flatten the image to 1d tensor
                images = images.reshape(-1, 28 * 28).to(self.device)
                # feed forward through the network
                prediction_layer = self.forward(Layer(images, 0))
                # calculate the loss
                loss = self.softmax_loss(prediction_layer.layer,
                                         labels.to(self.device))
                # backpropagate through the network
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # save data for plots
                self.loss_plot.append(i)
                self.loss_plot.append(loss.item())
            print(loss)
            # create a graph of the loss in perspective to the iterations
            graph = np.array(self.loss_plot).\
                reshape(int(len(self.loss_plot)/2), 2)
            if epoch == 1:
                self.viz.scatter(graph)
            # zero the data
            self.loss_plot = []
            # every epoch calculate the averge loss
            self.cost_plot.append(epoch)
            self.cost_plot.append(np.mean(graph[1]))
        # create a graph of the cost in respect to the ephoces
        graph = np.array(self.cost_plot).reshape(
            int(len(self.cost_plot) / 2), 2)
        self.viz.scatter(graph)

    def forward(self, layer):
        """
               this is the forward iteration of the network
               :param layer: a layer object
               :return: the prediction layer
               """
        if layer.index == len(self.weights):
            return layer
        else:
            new_layer = self.create_activated_layer(layer)
            layer_1 = self.forward(new_layer)
            return layer_1

    def create_activated_layer(self, layer):
        """
        creates activated layer using the relu function
        :param layer: layer objects
        :return: the next layer of the network
        """
        # create the layer of the model
        calculated_layer = layer.layer.matmul(
            self.weights[layer.index].weight.t()) + self.weights[layer.index].\
            bias
        # perform the activation function on every neuron [relu]
        # don't activate the prediction layer
        if layer.index < len(self.weights) - 1:
            activated_layer = calculated_layer.clamp(min=0)
        else:
            activated_layer = calculated_layer
        return Layer(layer_tensor=activated_layer,
                     layer_number=layer.index + 1)

    @staticmethod
    def softmax_loss(prediction_layer, expected_output):
        """
        calculate the nll loss of the network
        :param prediction_layer: the predication layer
        :param expected_output: the loss of the model
        :return:the loss
        """
        # softmax + log is better than performing the actions separately
        loss_softmax = torch.nn.CrossEntropyLoss()
        loss = loss_softmax(prediction_layer, expected_output)
        return loss

    def init_weights_linear_profile(self, sizes):
        """
        this profile creates the weights of the network
        :param sizes: the size of the network
        :return:
        """
        for index in range(self.network_layers):
            self.weights.append(torch.nn.Linear(
                sizes[index][0], sizes[index][1]).to(self.device))

    def test_model(self, test_data):
        total = 0
        correct = 0
        cost = []
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_data):
                images = images.reshape(-1, 28 * 28).to(self.device)
                prediction_layer = self.forward(Layer(images, 0)).layer
                _, predicted = torch.max(prediction_layer.data, 1)
                labels = labels.to(torch.device('cuda'))
                total += labels.size(0)
                cost.append(i)
                cost.append(self.softmax_loss(prediction_layer, labels).item())
                correct += (predicted == labels).sum().item()
            graph = np.array(cost).reshape(int(len(cost) / 2), 2)
            self.viz.scatter(graph)
            print(correct / 100)


def load_model(path, vis, layers=4):
    model = NeuralNetwork(vis, layers)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def main():
    vis = visdom.Visdom()
    train = torchvision.datasets.mnist.MNIST(
        r"..\..\data", train=True, download=True,
        transform=torchvision.transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset=train,
                                               batch_size=100,
                                               shuffle=True)
    test_dataset = torchvision.datasets.MNIST(root=r'..\..\data',
                                              train=False,
                                              transform=torchvision.transforms.
                                              ToTensor())

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=1,
                                              shuffle=False)

    evalute_network(vis, train_loader, test_loader)


def evalute_network(vis, train_loader, test_loader):
    n = NeuralNetwork(vis, 4)
    n.train_model(800, train_loader)
    n.test_model(test_loader)
    torch.save(n.state_dict(), 'n.ckpt')


if __name__ == '__main__':
    main()
