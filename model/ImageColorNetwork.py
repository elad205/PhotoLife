import numpy as np
import torch
import visdom
import torchvision.datasets.mnist
import torchvision
import copy
from data.DataParser import DataParser
from torch import nn

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


class PreTrainedModel(object):
    def __init__(self):
        self.modified_res_net = torchvision.models.resnet18(pretrained=True).\
            to('cuda')
        self.modified_res_net.conv1.weight =  \
            torch.nn.Parameter(self.modified_res_net.conv1.weight.sum(dim=1).
                               unsqueeze(1).data)

        self.proccesed_features = \
            torch.nn.Sequential(*list(self.modified_res_net.children())[0:6])

    def return_resnet_output(self, bw_image):
        return self.proccesed_features(bw_image)


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
        kernal = self.net.weights[self.index].kernel_size

        if type(kernal) == int:
            kernal = (kernal, kernal)

        self.net.weights[self.index].padding = (kernal[0] // 2, kernal[1] // 2)


class NeuralNetwork(nn.Module):
    def __init__(self, viz_tool, layer_number, pre_model):
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
            ('conv', 128, 128, 3, 1),
            ('batchnorm', 128),
            ('relu', None),
            ('decoder', 2),
            ('conv', 128, 64, 3, 1),
            ('batchnorm', 64),
            ('relu', None),
            ('conv', 64, 64, 3, 1),
            ('relu', None),
            ('decoder', 2),
            ('conv', 64, 32, 3, 1),
            ('tanh', None),
            ('decoder', 2)])

        # create an optimizer for the network
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.rate,
                                         momentum=0.8)
        # create the plots for debugging
        self.cost_plot = Plot("epochs", "cost", self.viz)
        self.accuracy_plot = Plot("epochs", "accuracy",  self.viz)

        self.pre_model = pre_model

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
        # if there is no eval phase
        for epoch in range(epochs):
            # run on all the data examples parsed by pytorch vision
            for i, (images, labels) in enumerate(train_data):
                # feed forward through the network
                images = images.to(self.device)
                labels = labels.to(self.device)
                images = self.pre_model.return_resnet_output(images)
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
        self.cost_plot.draw_plot(self.test_logger, "test" + serial)

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
        # perform the activation function on every neuron [relu]
        # the last layer has to be with negative values so we use tanh

        layer_params = self.weights[layer.index]

        # perform same padding on conv layers
        if type(layer_params) is nn.Conv2d or type(layer_params) is \
                nn.ConvTranspose2d:
            layer.calc_same_padding()

        # pass the data through
        calculated_layer = layer_params(layer.layer)

        # return the next layer in the network
        return Layer(layer_tensor=calculated_layer,
                     layer_number=layer.index + 1, net=self)

    @staticmethod
    def mse_loss(prediction_layer, expected_output):
        loss_mse = nn.MSELoss()
        loss = loss_mse(prediction_layer, expected_output)
        return loss

    def init_weights_liniar_conv(self, sizes):
        for index in range(self.network_layers):
            if sizes[index][0] == 'lin':
                self.weights.append(nn.Linear(
                    sizes[index][1], sizes[index][2]).to(self.device))

            if sizes[index][0] == 'conv':
                self.weights.append(nn.Conv2d(
                    sizes[index][1], sizes[index][2],
                    sizes[index][3], stride=sizes[index][4]).to(self.device))

            if sizes[index][0] == "decoder":
                self.weights.append(nn.Upsample(
                    scale_factor=sizes[index][1])).to(self.device)

            if sizes[index][0] == "batchnorm":
                self.weights.append(nn.BatchNorm2d(sizes[index][1]))\
                    .to(self.device)

            if sizes[index][0] == "deconv":
                self.weights.append(nn.ConvTranspose2d(
                    sizes[index][1], sizes[index][2], sizes[index][3],
                    stride=sizes[index][4])).to(self.device)

            if sizes[index][0] == "relu":
                self.weights.append(nn.ReLU())

            if sizes[index][0] == "tanh":
                self.weights.append(nn.Tanh())

    def test_model(self, test_data, display_data=False):
        """
        this function tests the model, it iterates on the testing data and
        feeds it to the network without backprop
        :param test_data: the testing data
        :param display_data: wether or not the data will be displayed
        on the visdom server
        :return:
        """
        viz_win_images = None
        viz_win_res = None
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_data):
                # display the images
                if display_data:
                    disp = DataParser.reconstruct_image(images, labels)
                    if viz_win_images is None:
                        viz_win_images = self.viz.images(disp)
                    else:
                        self.viz.images(disp, win=viz_win_images)

                    images_copy = copy.copy(images)

                # parse the images and labels
                images = images.to(self.device)
                labels = labels.to(self.device)
                images = self.pre_model.return_resnet_output(images)
                # feed forward through the network
                prediction_layer = self.forward(Layer(images, 0, net=self)).\
                    layer

                if display_data:
                    final = DataParser.reconstruct_image(images_copy,
                                                         prediction_layer)
                    if viz_win_res is None:
                        viz_win_res = self.viz.images(final)
                    else:
                        self.viz.images(final, win=viz_win_res)
                # calculate right answers
                # time.sleep(2)
                self.test_logger["loss"].append(self.mse_loss(
                    prediction_layer, labels).item())

            self.test_logger["cost"].append(np.mean(self.test_logger["loss"]))
            self.test_logger["loss"] = []


def load_model(path, vis, layers):
    """
    loads the model from .ckpt file
    :param path: the path of the file
    :param vis: vis object to display data
    :param layers: the number of layers of the network
    :return: loaded model
    """
    pre_model = PreTrainedModel()
    model = NeuralNetwork(vis, layers, pre_model)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def main():
    # connect to the visdom server
    vis = visdom.Visdom()
    train_data = DataParser()
    test_data = DataParser()
    # initialise data set
    train_loader, test_loader = DataParser.load_places_dataset(batch_size=16)
    train_data.parse_data(train_loader, stopper=938)
    test_data.parse_data(test_loader, stopper=130)
    # create, train and test the network
    create_new_network(vis, train_data, test_data,  layers=13, epochs=50)


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
    print("started training")
    pre_model = PreTrainedModel()
    model = NeuralNetwork(vis, layers, pre_model)
    model.train_model(epochs, train_loader, '1', test_loader)
    model.test_model(test_loader, display_data=True)
    torch.save(model.state_dict(), 'colorizer.ckpt')
    return model


if __name__ == '__main__':
    main()
