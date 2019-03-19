import numpy as np
import torch
import visdom
import torchvision.datasets.mnist
import torchvision
from data.DataParser import DataParser
from torch import nn

"""
File Name       :  ImageColorNetwork.py
Author:         :  Elad Cynamon
Date            :  11.02.2019
Version         :  1.0 ready

this file is a nural net model that colorizes black and white images.
the first input is passed through the pre-trained resnet18 model and after 
the few first layers to the colorizer network.
the module used for the network is pytorch.
the program displays graphs and test data on a visdom server.
note that in order to run the file you need to open a visdom server.
the data set which is currently used is the places dataset.
"""


class IterSetpScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        super(IterSetpScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.gamma ** (self.last_epoch / self.step_size)
                for base_lr in self.base_lrs]

    def update(self):
        self.last_epoch += 1


class PreTrainedModel(object):
    def __init__(self):
        """
        this function loads the resnet model and changes its input to fit
        black and white images.
        """
        self.modified_res_net = torchvision.models.resnet18(pretrained=True).\
            to('cuda')
        self.modified_res_net.conv1.weight =  \
            torch.nn.Parameter(self.modified_res_net.conv1.weight.sum(dim=1).
                               unsqueeze(1).data)

        self.proccesed_features = \
            torch.nn.Sequential(*list(self.modified_res_net.children())[0:6])

    def return_resnet_output(self, bw_image):
        """
        passes the image through the model
        :param bw_image: a black and white image
        :return: the output of the first six layers
        """
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
        layer_dim = self.layer.size(2)
        stride = self.net.weights[self.index].stride

        if type(kernal) == int:
            kernal = (kernal, kernal)

        if type(stride) == tuple:
            stride = stride[0]

        out_dim = (layer_dim + stride - 1) // stride

        padding = max(0, (out_dim - 1) * stride + (kernal[0]) - layer_dim)

        if padding % 2 != 0:
            self.layer = torch.nn.functional.pad(
                self.layer, [0, int(True), 0, int(True)])
        self.net.weights[self.index].padding = (padding // 2, padding // 2)


class NeuralNetwork(nn.Module):
    def __init__(self, viz_tool, pre_model, learnin_rate,
                 structure):
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
            self.rate = learnin_rate
            # create loggers for the training process and the testing process.
            self.train_logger = {"loss": [], "cost": [], "epochs": []}

            self.test_logger = {"loss": [], "epochs": [], "cost": []}
            # the visdom server
            self.viz = viz_tool
            self.network_layers = len(structure)
            # this is a spacial list of the weights, it acts like a normal
            # list but it contains autograd objects
            self.weights = torch.nn.ModuleList()

        # initiate the weights
        self.init_weights_liniar_conv(structure)

        # create an optimizer for the network
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4,
                                          betas=(0.5, 0.999))

        self.scheduler = IterSetpScheduler(self.optimizer, 1e5, gamma=0.1)

        # create the plots for debugging
        self.cost_plot = Plot("epochs", "cost", self.viz)
        self.accuracy_plot = Plot("epochs", "accuracy",  self.viz)

        self.pre_model = pre_model

        self.decoder_info = {}

    def train_model(self, images, labels, discriminator=None):
        """
        trains the model, this function takes the training data and preforms
        the feed forward, back propagation and the optimisation process
        :return:
        """

        # feed forward through the network
        prediction_layer = self.forward(
            Layer(images, 0, net=self), True).layer
        self.decoder_info.clear()
        # calculate the loss
        processed = torch.cat((prediction_layer, images), 1)
        decision_dis = discriminator.forward(
            Layer(processed, 0, net=discriminator)).layer

        x = torch.ones(decision_dis.size()).to(self.device)
        loss = self.bce_loss(torch.squeeze(decision_dis),
                             torch.squeeze(x))

        l1_loss = self.l1_loss(prediction_layer, labels)

        comb_loss = loss + 100 * l1_loss
        # backpropagate through the network
        self.optimizer.zero_grad()
        comb_loss.backward()
        self.optimizer.step()
        # save data for plots
        self.train_logger["loss"].append(comb_loss.item())

    def forward(self, layer, encoder_flag=False):
        """
        this is the forward iteration of the network.
        this function is recursive and uses the create activated layer function
        :param layer: a layer object
        :param encoder_flag: a flag which indicates if a u-net structure
        is present.
        :return: the prediction layer
        """
        if layer.index == len(self.weights):
            return layer
        else:
            new_layer = self.create_activated_layer(layer, encoder_flag)
            layer_1 = self.forward(new_layer, encoder_flag=encoder_flag)
            return layer_1

    def create_activated_layer(self, layer, encoder_flag):
        """
        creates activated layer using the relu function
        :param layer: layer objects
        :param encoder_flag: a flag which indicates if a u-net structure
        is present.
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

        if type(layer_params) is nn.LeakyReLU and encoder_flag:
            self.decoder_info[calculated_layer.size()] = calculated_layer

        if type(layer_params) is nn.ReLU and encoder_flag:
            calculated_layer = torch.cat(
                (calculated_layer, self.decoder_info[
                    calculated_layer.size()]), 1)

        return Layer(layer_tensor=calculated_layer,
                     layer_number=layer.index + 1, net=layer.net)

    @staticmethod
    def mse_loss(prediction_layer, expected_output):
        loss_mse = nn.MSELoss()
        loss = loss_mse(prediction_layer, expected_output)
        return loss

    @staticmethod
    def l1_loss(prediction_layer, expected_output):
        loss = nn.L1Loss()
        loss = loss(prediction_layer, expected_output)
        return loss

    @staticmethod
    def bce_loss(prediction_layer, expected_output):
        loss = nn.BCEWithLogitsLoss()
        loss = loss(prediction_layer, expected_output)
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

            if sizes[index][0] == 'deconv':
                self.weights.append(nn.ConvTranspose2d(
                    sizes[index][1], sizes[index][2],
                    sizes[index][3], stride=sizes[index][4]).to(self.device))

            if sizes[index][0] == "decoder":
                self.weights.append(nn.Upsample(
                    scale_factor=sizes[index][1])).to(self.device)

            if sizes[index][0] == "batchnorm":
                self.weights.append(nn.BatchNorm2d(sizes[index][1]))\
                    .to(self.device)

            if sizes[index][0] == "relu":
                self.weights.append(nn.ReLU())

            if sizes[index][0] == "tanh":
                self.weights.append(nn.Tanh())

            if sizes[index][0] == "leaky":
                self.weights.append(nn.LeakyReLU(sizes[index][1]))

    def test_model(self, test_data, images_per_epoch, display_data=False):
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
        viz_win_images = None
        viz_win_res = None
        with torch.no_grad():
            for i, (images, _) in enumerate(test_data):
                if (16 * i) >= images_per_epoch:
                    break
                images_gray, labels_lab = DataParser.convert_to_lab(images)
                # display the images
                if display_data:
                    disp = images
                    if viz_win_images is None:
                        viz_win_images = self.viz.images(disp)
                    else:
                        self.viz.images(disp, win=viz_win_images)

                # parse the images and labels
                images_gray = torch.from_numpy(images_gray).to(self.device)
                labels = torch.from_numpy(labels_lab).to(self.device)
                # feed forward through the network
                prediction_layer = self.forward(Layer(images_gray, 0, net=self
                                                      ), True).layer

                if display_data:
                    final = prediction_layer
                    if viz_win_res is None:
                        viz_win_res = self.viz.images(
                            DataParser.convert_to_lab(final.cpu(), False))
                    else:
                        self.viz.images(final, win=viz_win_res)

                self.test_logger["loss"].append(self.mse_loss(
                    prediction_layer, labels).item())

            self.test_logger["cost"].append(np.mean(self.test_logger["loss"]))
            self.test_logger["loss"] = []


class Discriminator(NeuralNetwork):
    def __init__(
            self, viz_tool, pre_model, learnin_rate, structure,
            generator):
        super(Discriminator, self).__init__(
            viz_tool, pre_model, learnin_rate, structure)
        self.generator = generator
        self.cost_plot_discrim = Plot("epochs", "cost", self.viz)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4 / 10,
                                          betas=(0.5, 0.999))

        self.scheduler = IterSetpScheduler(self.optimizer, 1e5)

    def train_model(self, images, labels, discriminator=None):
            processed = torch.cat((labels, images), 1)
            real_res = self.forward(Layer(
                processed, 0, net=self)).layer
            x = torch.ones(real_res.size()) * 0.9
            x = x.to(self.device)
            self.optimizer.zero_grad()
            loss_real = self.bce_loss(torch.squeeze(real_res),
                                      torch.squeeze(x))
            loss_real.backward()
            fake_res = self.generator(Layer(images, 0, net=self.generator),
                                      True). \
                layer
            self.generator.decoder_info.clear()

            fake_processed = torch.cat((fake_res, images), 1)

            fake_res = self.forward(Layer(fake_processed, 0, net=self)).layer
            x = torch.zeros(fake_res.size()).to(self.device)
            loss_fake = self.bce_loss(torch.squeeze(fake_res), torch.squeeze(x)
                                      )
            loss_fake.backward()

            self.optimizer.step()

            self.train_logger["loss"].append(loss_fake.item()
                                             + loss_real.item())


class CombinedTraining(object):
    def __init__(self, discriminator):
        self.discriminator = discriminator

    def super_train(self, epochs, train_loader, serial, images_per_epoch,
                    test_data=None):
        for epoch in range(epochs):

            prog = DataParser.create_loading_bar(train_loader,
                                                 images_per_epoch / 16)

            if self.discriminator.generator.scheduler.get_lr()[0] > 1e-6:
                self.discriminator.scheduler.step()
                self.discriminator.generator.scheduler.step()

            for i, (images_rgb, _) in enumerate(train_loader):

                if (16 * i) >= images_per_epoch:
                    break

                self.discriminator.generator.scheduler.update()
                self.discriminator.scheduler.update()

                with torch.no_grad():
                    images, labels = DataParser.convert_to_lab(images_rgb)
                    images = torch.from_numpy(images).to(
                        self.discriminator.device)
                    labels = torch.from_numpy(labels).to(
                        self.discriminator.device)

                # train the discriminator
                self.discriminator.train_model(images, labels)

                # train the generator
                self.discriminator.generator.train_model(
                    images, labels, discriminator=self.discriminator)

                prog.update(i * 16)

            prog.finish()
            print(
                "epoch {0}\n avg loss of discriminator is {1}\n"
                " avg loss of generator {2}".format(
                    epoch, np.mean(self.discriminator.train_logger["loss"]),
                    np.mean(
                        self.discriminator.generator.train_logger["loss"])))
            # every epoch calculate the average loss

            self.discriminator.train_logger["cost"].append(np.mean(
                self.discriminator.train_logger["loss"]))

            self.discriminator.generator.train_logger["cost"].append(np.mean(
                self.discriminator.generator.train_logger["loss"]))
            # zero out the losss
            self.discriminator.train_logger["loss"] = []
            self.discriminator.generator.train_logger["loss"] = []

        if test_data is not None:
            self.discriminator.generator.test_model(test_data, 112, True)
            # add the number of epochs that were done

        self.discriminator.train_logger["epochs"] = list(range(epochs))
        self.discriminator.generator.train_logger["epochs"] = \
            list(range(epochs))
        self.discriminator.test_logger["epochs"] = list(range(epochs))
        self.discriminator.generator.test_logger["epochs"] = \
            list(range(epochs))
        # create a graph of the cost in respect to the epochs
        self.discriminator.cost_plot.draw_plot(
            self.discriminator.train_logger, "train" + serial)
        self.discriminator.generator.cost_plot.draw_plot(
            self.discriminator.generator.train_logger, "train" + serial)
        # self.discriminator.cost_plot.draw_plot
        # (self.discriminator.test_logger, "test" + serial)
        # self.discriminator.generator.cost_plot.draw_plot(
        # self.discriminator.generator.test_logger, "test" + serial)
        # zero the loggers
        self.discriminator.train_logger["cost"] = []
        self.discriminator.generator.train_logger["cost"] = []
        self.discriminator.test_logger["cost"] = []
        self.discriminator.generator.test_logger["cost"] = []


def load_model(path, vis):
    """
    loads the model from .ckpt file
    :param path: the path of the file
    :param vis: vis object to display data
    :return: loaded model
    """
    pre_model = PreTrainedModel()
    model = NeuralNetwork(vis, pre_model, 0.1, [
            ('conv', 1, 64, 3, 2),
            ('batchnorm', 64),
            ('leaky', 0.1),
            ('conv', 64, 128, 3, 2),
            ('batchnorm', 128),
            ('leaky', 0.1),
            ('conv', 128, 256, 3, 2),
            ('batchnorm', 256),
            ('leaky', 0.1),
            ('conv', 256, 512, 3, 2),
            ('batchnorm', 512),
            ('leaky', None),
            ('deconv', 512, 512, 3, 2),
            ('batchnorm', 512),
            ('relu', None),
            ('deconv', 512, 256, 3, 2),
            ('batchnorm', 256),
            ('relu', None),
            ('deconv', 256, 128, 3, 2),
            ('batchnorm', 128),
            ('relu', None),
            ('deconv', 128, 64, 3, 2),
            ('batchnorm', 64),
            ('relu', None),
            ('deconv', 64, 3, 3, 2),
            ('tanh', None)])
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def main(train_flag=True):
    # connect to the visdom server
    vis = visdom.Visdom()
    print("make sure visdom server is activated")

    # initialise data set
    train_loader, test_loader = DataParser.load_places_dataset(batch_size=16)
    # create, train and test the network
    gen = create_new_network(vis, train_loader, test_loader)
    torch.save(gen.state_dict(), 'generator.ckpt')
    if not train_flag:
        # load the model and test if
        model = load_model("colorizer.ckpt", vis)
        return model


def create_new_network(vis, train_loader, test_loader):
    """
    this function initialise the network, trains it on the train data and
    the evaluation data, tests it on the test data and saves the weights.
    :param vis: the visdom server to display graphs
    :param train_loader: the training data
    :param test_loader: the testing data
    :return: the model created
    """
    print("started training")
    pre_model = PreTrainedModel()
    model = NeuralNetwork(vis, pre_model, 0.1, [
            ('conv', 1, 64, 4, 1),
            ('leaky', 0.2),
            ('conv', 64, 64, 4, 2),
            ('leaky', 0.2),
            ('conv', 64, 128, 4, 2),
            ('leaky', 0.2),
            ('conv', 128, 256, 4, 2),
            ('leaky', 0.2),
            ('conv', 256, 512, 4, 2),
            ('leaky', 0.2),
            ('conv', 512, 512, 4, 2),
            ('leaky', 0.2),
            ('conv', 512, 512, 4, 2),
            ('leaky', 0.2),
            ('conv', 512, 512, 4, 2),
            ('leaky', 0.2),
            ('deconv', 512, 512, 4, 2),
            ('relu', None),
            ('deconv', 1024, 512, 4, 2),
            ('relu', None),
            ('deconv', 1024, 512, 4, 2),
            ('relu', None),
            ('deconv', 1024, 256, 4, 2),
            ('relu', None),
            ('deconv', 512, 128, 4, 2),
            ('relu', None),
            ('deconv', 256, 64, 4, 2),
            ('relu', None),
            ('deconv', 128, 64, 4, 2),
            ('relu', None),
            ('conv', 128, 3, 1, 1),
            ('tanh', None)])

    model2 = Discriminator(vis, None, 0.1, [('conv', 4, 64, 4, 2),
                                            ('batchnorm', 64),
                                            ('leaky', 0.2),
                                            ('conv', 64, 128, 4, 2),
                                            ('batchnorm', 128),
                                            ('leaky', 0.2),
                                            ('conv', 128, 256, 4, 2),
                                            ('batchnorm', 256),
                                            ('leaky', 0.2),
                                            ('conv', 256, 512, 4, 1),
                                            ('batchnorm', 512),
                                            ('leaky', 0.2),
                                            ('conv', 512, 1, 4, 1)
                                            ], model)

    trainer = CombinedTraining(model2)
    trainer.super_train(30, train_loader, '1', 1000, test_loader)

    return model


if __name__ == '__main__':
    # set train flag to false to load pre trained model
    main(train_flag=True)
