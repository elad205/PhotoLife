from abc import ABC, abstractmethod
from torch import nn
import torch
from colorization.loggers import Plot
from torch.nn.utils.spectral_norm import spectral_norm
from colorization.layers import LayerTypes, Layer
from visdom import Visdom


class IterSetpScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        super(IterSetpScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * 10 for base_lr in self.base_lrs]


class NeuralNetwork(nn.Module, ABC):
    def __init__(self, viz_tool: Visdom, learning_rate: tuple, structure: list,
                 optimizer: torch.optim.Optimizer):
        """
        initialize the network variables
        :param viz_tool: visdom object to display plots
        :param layer_number: the number of layers in the network
        """
        super(NeuralNetwork, self).__init__()
        # using the nvidia CUDA api in order to perform calculations on a
        # GPU
        self.device = torch.device('cuda' if torch.cuda.is_available()
                                   else 'cpu')
        # the learning rate of the model
        self.rate = learning_rate
        # create loggers for the training process and the testing process.
        self.train_logger = {"loss": [], "cost": [], "epochs": []}

        self.test_logger = {"loss": [], "epochs": [], "cost": []}
        # the visdom server
        self.viz = viz_tool
        # this is a spacial list of the weights, it acts like a normal
        # list but it contains autograd objects
        self.weights = torch.nn.ModuleList()

        self.structure = structure
        # create an optimizer for the network
        self.optimizer = optimizer

        # create the plots for debugging
        self.cost_plot = Plot("epochs", "cost", self.viz)

        self.accuracy_plot = Plot("epochs", "accuracy",  self.viz)

        self.dict_layers = {
            "conv": LayerTypes.conv_layer,
            "relu": LayerTypes.relu_layer,
            "batchnorm": LayerTypes.batchnorm_layer,
            "lin": LayerTypes.liniar_layer
        }

    @abstractmethod
    def train_model(self, images: torch.Tensor, labels: torch.Tensor,
                    extra_net):
        """
        trains the model, this function takes the training data and preforms
        the feed forward, back propagation and the optimisation process
        :return:
        """
        pass

    @abstractmethod
    def forward(self, layer: Layer):
        """
        this is the forward iteration of the network.
        this function is recursive and uses the create activated layer function
        :param layer: a layer object
        is present.
        :return: the prediction layer
        """
        pass

    def create_activated_layer(self, layer: Layer):
        """
        creates activated layer using the relu function
        :param layer: layer objects
        :return: the next layer of the network a Layer object with the next
        index
        """
        layer_params = self.weights[layer.index]
        # perform same padding on conv layers
        if type(layer_params) is nn.Conv2d or type(layer_params) is \
                nn.ConvTranspose2d:
            layer.calc_same_padding()
        # pass the data through

        # return the next layer in the network

        calculated_layer = layer_params(layer.layer)

        return Layer(layer_tensor=calculated_layer,
                     layer_number=layer.index + 1, net=layer.net)

    @abstractmethod
    def test_model(
            self, test_data:
            torch.utils.data.DataLoader,
            images_per_epoch: int, batch_size: int, display_data=False):
        """
        this function tests the model, it iterates on the testing data and
        feeds it to the network without backprop
        :param test_data: the testing data
        :param images_per_epoch: to test on small batches of dataset this
        param indicates how many images to iterate per epoch.
        :param display_data: wether or not the data will be displayed
        on the visdom server
        :return:
        """
        pass

    def basic_switch(self, layer: tuple):
        return self.dict_layers.get(layer[0], None)(layer)

    def init_weights(self, init_func=spectral_norm):
        network_layers = []
        for layer_index in range(len(self.structure)):
            self.structure[layer_index] += (init_func, )
            layer = self.basic_switch(self.structure[layer_index])
            if type(layer) is list:
                for sub_layer in layer:
                    network_layers.append(sub_layer)
            else:
                network_layers.append(layer)

        return network_layers

    def forward_model(self, layer: Layer):
        """
        this is the forward iteration of the network.
        this function is recursive and uses the create activated layer function
        :param layer: a layer object
        is present.
        :return: the prediction layer
        """
        if layer.index == len(self.weights):
            return layer
        else:
            new_layer = self.create_activated_layer(layer)
            layer_1 = self.forward_model(new_layer)
            return layer_1
