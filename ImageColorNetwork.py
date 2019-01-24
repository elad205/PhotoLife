import numpy as np
import torch
import visdom
import torchvision.datasets.mnist
import time
import copy
from DataParser import DataParser

"""
File Name       :  MnistModel.py
Author:         :  Elad Cynamon
Date            :  11.01.2018
Version         :  1.0 ready

this file is a Neural network model that classifies digits and trains on 
the mnist dataset. 
the module used for the network is pytorch.
the program displays graphs and test data on a visdom server.
note that in order to run the file you need to open a visdom server. 
"""


class Plot:
    def __init__(self, name_x, name_y, viz):
        """
        this class represents a visdom plot. It contains the name of the x axis
        and the y axis which define the type of the plot
        :param name_x: the name of the x axis
        :param name_y: the name of the y axis
        :param viz: the visdom server object
        """
        self.x_title = name_x
        self.y_title = name_y
        self.viz = viz
        self.window = None

    def draw_plot(self, dict_vals, name, up='insert'):
        """
        this function sends the data of the plot to the visdom server.
         It takes a dictionary with the required values and extracts the
        :param dict_vals:
        :param name: the name of the line
        :param up: the type of update to perform to the graph
        :return: display the graph on the visdom server
        """
        # if there is no graph displayed than create a new graph
        if self.window is None:
            window = self.viz.line(
                X=dict_vals[self.x_title], Y=dict_vals[self.y_title],
                name=name, opts=dict(xlabel=self.x_title, ylabel=self.y_title))
            self.window = window
        # if there is already a graph than append the line to the existing
        # graph
        else:
            self.viz.line(X=dict_vals[self.x_title], Y=dict_vals[self.y_title],
                          name=name, win=self.window,
                          update=up, opts=dict(
                    xlabel=self.x_title, ylabel=self.y_title))


class Layer:
    """
    this class represents a layer in the network, it contains the
    layer tensor and the place of the layer in the network
    """
    def __init__(self, layer_tensor, layer_number, net):
        self.layer = layer_tensor
        self.index = layer_number
        self.net = net
        try:
            # give the input layer of the linear layer which comes after
            # the conv layer
            if type(net.weights[self.index]) is torch.nn.Linear and type(
                    net.weights[self.index - 1]) is torch.nn.Conv2d:
                self.layer = self.layer.view(-1, net.weights[self.index].
                                             in_features)
        except IndexError:
            pass

    def calc_same_padding(self):

        kernel_of_next_conv = self.net.weights[self.index].kernel_size

        if type(kernel_of_next_conv) is tuple:
            kernel_of_next_conv = kernel_of_next_conv[0]

        padding_size = int((kernel_of_next_conv - 1) / 2)
        self.net.weights[self.index].padding = (padding_size, padding_size)


class NeuralNetwork(torch.nn.Module):
    def __init__(self, viz_tool, layer_number):
        """
        initialize the network variables
        :param viz_tool: visdom object to display plots
        :param layer_number: the number of layers in the network
        """
        super(NeuralNetwork, self).__init__()
        with torch.no_grad():
            # using the nvidia CUDA api in order to perform calculations on a
            # GPU
            self.device = torch.device('cuda' if torch.cuda.is_available()
                                       else 'cpu')
            # the learning rate of the model
            self.rate = 0.1
            # create loggers for the training process and the testing process.
            self.train_logger = {"loss": [], "cost": [], "epochs": []}

            self.test_logger = {"loss": [], "epochs": [], "cost": []}
            # the visdom server
            self.viz = viz_tool
            self.network_layers = layer_number
            # this is a spacial list of the weights, it acts like a normal
            # list but it contains autograd objects
            self.weights = torch.nn.ModuleList()

        # initiate the weights
        self.init_weights_liniar_conv([
            ('conv', 1, 64, 3),
            ('maxpooling', 3, 2),
            ('conv', 64, 128, 3),
            ('maxpooling', 3, 2),
            ('conv', 128, 256, 3),
            ('maxpooling', 3, 2),
            ('conv', 256, 512, 3),
            ('conv', 512, 256, 3),
            ('decoder', 256, 128, 2, 2),
            ('decoder', 128, 64, 2, 2),
            ('conv', 64, 32, 3),
            ('decoder', 32, 2, 2, 2)])

        # create an optimizer for the network
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.rate,
                                         momentum=0.8)
        # create the plots for debugging
        self.cost_plot = Plot("epochs", "cost", self.viz)
        self.accuracy_plot = Plot("epochs", "accuracy",  self.viz)

    def train_model(self, epochs, train_data, serial, test_data=None):
        """
        trains the model, this function takes the training data and preforms
        the feed forward, back propagation and the optimisation process
        :param epochs: the number of epochs to perform
        :param train_data: the data set
        :param serial: unique identifier of the plot
        :param test_data: the data to run tests on every epoch
        :return:
        """
        loss = None
        # if there is no eval phase
        for epoch in range(epochs):
            # run on all the data examples parsed by pytorch vision
            for i, (images, labels) in enumerate(train_data):
                # flatten the image to 1d tensor
                # images = images.reshape(-1, 28 * 28).to(self.device)
                # feed forward through the network
                images = images.to(self.device)
                labels = labels.to(self.device)
                prediction_layer = self.forward(Layer(images, 0, net=self))
                # calculate the loss
                loss = self.mse_loss(prediction_layer.layer, labels)

                # backpropagate through the network
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # save data for plots
                self.train_logger["loss"].append(loss.item())

            print("epoch {0} avg loss is {1}".format(
                epoch, np.mean(self.train_logger["loss"])))

            # every epoch calculate the average loss
            self.train_logger["cost"].append(np.mean(
                self.train_logger["loss"]))
            # zero out the losss
            self.train_logger["loss"] = []

            if test_data is not None:
                self.test_model(test_data)
        # add the number of epochs that were done
        self.train_logger["epochs"] = list(range(epochs))
        self.test_logger["epochs"] = list(range(epochs))
        # create a graph of the cost in respect to the epochs
        self.cost_plot.draw_plot(self.train_logger, "train" + serial)

        # zero the loggers
        self.train_logger["cost"] = []
        self.test_logger["cost"] = []

    def forward(self, layer):
        """
        this is the forward iteration of the network.
        this function is recursive and uses the create activated layer function
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
        :return: the next layer of the network a Layer object with the next
        index
        """
        # create the layer of the model
        if type(self.weights[layer.index]) is not torch.nn.ConvTranspose2d:
            layer.calc_same_padding()
        calculated_layer = self.weights[layer.index](layer.layer)
        # perform the activation function on every neuron [relu]
        # don't activate the prediction layer
        if layer.index < len(self.weights) - 1 and type(
                self.weights[layer.index]) is not torch.nn.MaxPool2d:
            activated_layer = calculated_layer.clamp(min=0)
        else:
            activated_layer = calculated_layer
        return Layer(layer_tensor=activated_layer,
                     layer_number=layer.index + 1, net=self)

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

    @staticmethod
    def mse_loss(prediction_layer, expected_output):
        loss_mse = torch.nn.MSELoss()
        loss = loss_mse(prediction_layer, expected_output)
        return loss

    def init_weights_liniar_conv(self, sizes):
        for index in range(self.network_layers):
            if sizes[index][0] == 'lin':
                self.weights.append(torch.nn.Linear(
                    sizes[index][1], sizes[index][2]).to(self.device))

            if sizes[index][0] == 'conv':
                self.weights.append(torch.nn.Conv2d(
                    sizes[index][1], sizes[index][2],
                    sizes[index][3]).to(self.device))

            if sizes[index][0] == "decoder":
                self.weights.append(
                    torch.nn.ConvTranspose2d(
                        sizes[index][1], sizes[index][2],
                        sizes[index][3], stride=sizes[index][4]).to(self.device))

            if sizes[index][0] == "maxpooling":
                self.weights.append(torch.nn.MaxPool2d(
                    kernel_size=sizes[index][1], stride=sizes[index][2]))

    def test_model(self, test_data, display_data=False):
        """
        this function tests the model, it iterates on the testing data and
        feeds it to the network without backprop
        :param test_data: the testing data
        :param display_data: wether or not the data will be displayed
        on the visdom server
        :return:
        """
        total = 0
        correct = 0
        viz_win_text = None
        viz_win_images = None
        if display_data:
            viz_win_text = self.viz.text("")
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_data):
                # display the images
                if display_data:
                    if viz_win_images is None:
                        viz_win_images = self.viz.images(images)
                    else:
                        self.viz.images(images, win=viz_win_images)

                # parse the images and labels
                # images = images.reshape(-1, 28 * 28).to(self.device)
                images = images.to(self.device)
                labels = labels.to(self.device)
                # feed forward through the network
                prediction_layer = self.forward(Layer(images, 0, net=self)).\
                    layer
                # check the most likely prediction in the layer
                _, predicted = torch.max(prediction_layer.data, 1)

                # display the answer
                if display_data:
                    data_display = str(predicted.cpu().numpy().tolist())
                    self.viz.text(data_display, win=viz_win_text)

                # calculate right answers
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # time.sleep(2)
                self.test_logger["loss"].append(self.softmax_loss(
                    prediction_layer, labels).item())
            # print the accuracy
            if display_data:
                self.viz.text(100 * correct / total)
                print("accuracy is " + str(100 * correct / total))

            self.test_logger["cost"].append(np.mean(self.test_logger["loss"]))
            self.test_logger["loss"] = []
            self.test_logger["accuracy"].append(correct / total)


def load_model(path, vis, layers):
    """
    loads the model from .ckpt file
    :param path: the path of the file
    :param vis: vis object to display data
    :param layers: the number of layers of the network
    :return: loaded model
    """
    model = NeuralNetwork(vis, layers)
    model.load_state_dict(torch.load(path))
    return model


def main():
    # connect to the visdom server
    vis = visdom.Visdom()
    train_data = DataParser()
    test_data = DataParser()
    # initialise data set
    train_loader, test_loader = DataParser.load_cifar10(batch_size=100)
    train_data.parse_data(train_loader)
    test_data.parse_data(test_loader)
    # create, train and test the network
    create_new_network(vis, train_data, test_data,  layers=12, epochs=10)

    # model = load_model("SGD_99.25.ckpt", vis, 4)
    # model.test_model(test_loader, display_data=True)


def create_new_network(vis, train_loader, test_loader, layers, epochs):
    """
    this function initialise the network, trains it on the train data and
    the evaluation data, tests it on the test data and saves the weights.
    :param vis: the visdom server to display graphs
    :param train_loader: the training data
    :param test_loader: the testing data
    :param layers: the number of layers in the model
    :param epochs: the number epochs to perform
    :return: the model created
    """
    model = NeuralNetwork(vis, layers)
    model.train_model(epochs, train_loader, '1')
    # model.test_model(test_loader, display_data=True)
    torch.save(model.state_dict(), 'model.ckpt')
    return model


if __name__ == '__main__':
    main()
