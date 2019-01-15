import numpy as np
import torch
import visdom
import math
import torchvision.datasets.mnist
import time

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


class Layer:
    """
    this class represents a layer in the network, it contains the
    layer tensor and the place of the layer in the network
    """
    def __init__(self, layer_tensor, layer_number, net):
        self.layer = layer_tensor
        self.index = layer_number
        try:
            if type(net.weights[self.index]) is torch.nn.Linear and type(net.weights[self.index -1]) is torch.nn.Conv2d:
                self.layer = self.layer.view(-1, 800)
        except IndexError:
            pass


class NeuralNetwork(torch.nn.Module):
    def __init__(self, viz_tool, layer_number):
        """
        initialize the network variables
        :param viz_tool: visdom object to display plots
        :param layer_number: the number of layers in the network
        """
        super(NeuralNetwork, self).__init__()
        # using the nvidia CUDA api in order to perform calculations on a GPU
        with torch.no_grad():
            self.device = torch.device('cuda' if torch.cuda.is_available()
                                       else 'cpu')
            self.rate = 0.0001
            self.loss_logger = []
            self.cost_logger = []
            self.accuracy_logger = []
            self.viz = viz_tool
            self.network_layers = layer_number
            self.weights = torch.nn.ModuleList()
            self.cost_window = None
            self.accuracy_window = None
        # initiate the weights
        self.init_weights_liniar_conv([('conv', 1, 20, 5), ('conv', 20, 50, 5), ('lin', 800, 500), ('lin', 500, 10)])

        # create an optimizer for the network
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.rate)

    def train_model(self, epochs, train_data):
        """
        trains the model, this function takes the training data and preforms
        the feed forward, back propagation and the optimisation process
        :param epochs: the number of epochs to perform
        :param train_data: the data set
        :return:
        """
        loss = None
        for epoch in range(epochs):
            counter = 0
            correct = 0
            total = 0
            # run on all the data examples parsed by pytorch vision
            for i, (images, labels) in enumerate(train_data):
                # flatten the image to 1d tensor
                # images = images.reshape(-1, 28 * 28).to(self.device)
                # feed forward through the network
                prediction_layer = self.forward(Layer(images, 0, net=self))
                # calculate the loss
                loss = self.softmax_loss(prediction_layer.layer,
                                         labels.to(self.device))

                # backpropagate through the network
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # save data for plots
                self.loss_logger.append(loss.item())
                counter += 1
                total += labels.size(0)

                # calculate the accuracy
                _, predicted = torch.max(prediction_layer.layer.data, 1)
                correct += (predicted == labels.to(self.device)).sum().item()

            print("epoch {0} avg loss is {1}".format(
                epoch, np.mean(self.loss_logger)))
            # create a graph of the loss in perspective to the iterations
            if epoch == 0:
                self.viz.line(X=list(range(counter)), Y=self.loss_logger)

            # zero the data
            # every epoch calculate the average loss
            self.cost_logger.append(np.mean(self.loss_logger))
            self.loss_logger = []
            self.accuracy_logger.append(correct / total)

        # create a graph of the cost in respect to the epochs
        if self.cost_window is None:
            self.cost_window = self.viz.line(
                X=list(range(epochs)), Y=self.cost_logger,
                name='train', opts=dict(xlabel='epoch', ylabel='cost'))
        else:
            self.viz.line(
                X=list(range(epochs)), Y=self.cost_logger,
                win=self.cost_window, update='append', name='eval')

        # create a graph of the accuracy in respect to epochs
        if self.accuracy_window is None:
            self.accuracy_window = self.viz.line(
                X=list(range(epochs)),
                Y=self.accuracy_logger, name='train',
                opts=dict(xlabel='epoch', ylabel='accuracy'))
        else:
            self.viz.line(X=list(range(epochs)),
                          Y=self.accuracy_logger, win=self.accuracy_window,
                          update='append', name='eval')

        # zero the loggers
        self.cost_logger = []
        self.accuracy_logger = []

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
        calculated_layer = self.weights[layer.index](layer.layer)
        # perform the activation function on every neuron [relu]
        # don't activate the prediction layer
        if layer.index < len(self.weights) - 1:
            activated_layer = calculated_layer.clamp(min=0)
            if type(self.weights[layer.index]) is torch.nn.Conv2d:
                activated_layer = torch.nn.functional.max_pool2d(activated_layer, 2, 2)
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

    def init_weights_linear_profile(self, sizes):
        """
        this profile creates the weights of the network
        :param sizes: the size of the network
        :return:
        """
        for index in range(self.network_layers):
            self.weights.append(torch.nn.Linear(
                sizes[index][0], sizes[index][1]).to(self.device))

    def init_weights_liniar_conv(self, sizes):
        for index in range(self.network_layers):
            if sizes[index][0] == 'lin':
                self.weights.append(torch.nn.Linear(
                    sizes[index][1], sizes[index][2]).to(self.device))

            if sizes[index][0] == 'conv':
                self.weights.append(torch.nn.Conv2d(sizes[index][1], sizes[index][2], sizes[index][3]))


    def test_model(self, test_data):
        """
        this function tests the model, it iterates on the testing data and
        feeds it to the network without backprop
        :param test_data: the testing data
        :return:
        """
        total = 0
        correct = 0
        viz_win_text = self.viz.text("")
        viz_win_images = None
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_data):
                # display the images
                if viz_win_images is None:
                    viz_win_images = self.viz.images(images)
                else:
                    self.viz.images(images, win=viz_win_images)

                # parse the images and labels
                #images = images.reshape(-1, 28 * 28).to(self.device)
                images = images.to(self.device)
                labels = labels.to(self.device)
                # feed forward through the network
                prediction_layer = self.forward(Layer(images, 0, net=self)).layer
                # check the most likely prediction in the layer
                _, predicted = torch.max(prediction_layer.data, 1)

                # display the answer
                data_display = str(predicted.cpu().numpy().tolist())
                self.viz.text(data_display, win=viz_win_text)

                # calculate right answers
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # time.sleep(2)

            # print the accuracy
            self.viz.text(100 * correct / total)
            print("accuracy is " + str(100 * correct / total))


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

    # initialise data set
    train_loader, eval_loader, test_loader = load_mnist(bath_size=100)
    # create, train and test the network
    create_new_network(vis, train_loader, test_loader, eval_loader, layers=4,
                       epochs=10)


def create_new_network(vis, train_loader, test_loader, eval_loader,
                       layers, epochs):
    """
    this function initialise the network, trains it on the train data and
    the evaluation data, tests it on the test data and saves the weights.
    :param vis: the visdom server to display graphs
    :param train_loader: the training data
    :param test_loader: the testing data
    :param eval_loader: the evaluation data
    :param layers: the number of layers in the model
    :param epochs: the number epochs to perform
    :return: the model created
    """
    model = NeuralNetwork(vis, layers)
    model.train_model(epochs, train_loader)
    model.train_model(epochs, eval_loader)
    model.test_model(test_loader)
    torch.save(model.state_dict(), 'model.ckpt')
    return model


def load_mnist(bath_size, train_size=0.8):
    """
    this function loads the mnist data set to the script.
    it splits the training data into train data and evaluation data
    by the ratio determined by train size, default is 80% - 20%.
    :param bath_size: the batch size of the data
    :param train_size: the size of the train data in relative to the entire
    train data
    :return: a train loader object, an eval loader and a test loader.
    """
    # load mnist data set
    train = torchvision.datasets.mnist.MNIST(
        "data", train=True, download=True,
        transform=torchvision.transforms.ToTensor())

    # split the data to train data and validation data
    # train is 80% of train data and validation is 20% of the train data
    total_train = int(len(train) * train_size)
    sampler = list(range(len(train)))
    train_data = sampler[0: total_train]
    eval_data = sampler[total_train:]
    # create a random sample for each data set
    train_data = torch.utils.data.sampler.SubsetRandomSampler(train_data)
    eval_data = torch.utils.data.sampler.SubsetRandomSampler(eval_data)

    # load training data
    train_loader = torch.utils.data.DataLoader(dataset=train,
                                               batch_size=bath_size,
                                               sampler=train_data)
    # load eval data
    eval_loader = torch.utils.data.DataLoader(
        dataset=train, batch_size=bath_size, sampler=eval_data)

    test_dataset = torchvision.datasets.MNIST(root='data',
                                              train=False,
                                              transform=torchvision.transforms.
                                              ToTensor())
    # load test data
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=bath_size,
                                              shuffle=False)

    return train_loader, eval_loader, test_loader


if __name__ == '__main__':
    main()
