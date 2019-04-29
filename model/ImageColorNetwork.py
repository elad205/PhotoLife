import numpy as np
from torch.nn.utils.spectral_norm import spectral_norm
import visdom
import torchvision.datasets.mnist
import torchvision
from data.DataParser import DataParser
from torch import nn
import tqdm
import cv2
import torch
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


class SelfAttention(nn.Module):
    def __init__(self, in_channel: int, gain: int = 1):
        super().__init__()
        self.query = self._spectral_init(nn.Conv1d(in_channel, in_channel // 8, 1), gain=gain).to('cuda')
        self.key = self._spectral_init(nn.Conv1d(in_channel, in_channel // 8, 1), gain=gain).to('cuda')
        self.value = self._spectral_init(nn.Conv1d(in_channel, in_channel, 1), gain=gain).to('cuda')
        self.gamma = nn.Parameter(torch.tensor(0.0)).to('cuda')

    @staticmethod
    def _spectral_init(module: nn.Module, gain: int = 1):
        nn.init.kaiming_uniform_(module.weight, gain)
        if module.bias is not None:
            module.bias.data.zero_()

        return nn.utils.spectral_norm(module)

    def forward(self, input: torch.Tensor):
        input = input.to('cuda')
        shape = input.shape
        flatten = input.view(shape[0], shape[1], -1)
        query = self.query(flatten).permute(0, 2, 1)
        key = self.key(flatten)
        value = self.value(flatten)
        query_key = torch.bmm(query, key)
        attn = nn.functional.softmax(query_key, 1)
        attn = torch.bmm(value, attn)
        attn = attn.view(*shape)
        out = self.gamma * attn + input
        return out


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()

        self.device = 'cuda'
        self.vgg_pre_trained = torchvision.models.vgg16(
            pretrained=True).features.to(self.device)
        self.blocks = []
        self.vgg_pre_trained.eval()
        for i in range(4):
            self.blocks.append(nn.Sequential().to(self.device))

        for i in range(4):
            self.blocks[0].add_module(str(i), self.vgg_pre_trained[i])

        for i in range(4, 9):
            self.blocks[1].add_module(str(i), self.vgg_pre_trained[i])

        for i in range(9, 16):
            self.blocks[2].add_module(str(i), self.vgg_pre_trained[i])

        for i in range(16, 23):
            self.blocks[3].add_module(str(i), self.vgg_pre_trained[i])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, image):
        with torch.no_grad():
            if type(image) is not torch.Tensor:
                image = torch.from_numpy(image).to(self.device)
            else:
                image = image.to(self.device)

            outputs = []
            out = image
            for block in self.blocks:
                outputs.append(block.forward(out))
                out = outputs[-1].to(self.device)

            return outputs


class IterSetpScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        super(IterSetpScheduler, self).__init__(optimizer, last_epoch)

    """"
    def get_lr(self):
        return [base_lr * self.gamma ** (self.last_epoch / self.step_size)
                for base_lr in self.base_lrs]
    """

    def get_lr(self):
        return [base_lr * 10 for base_lr in self.base_lrs]

    def update(self):
        self.last_epoch += 1


class PreTrainedModel(nn.Module):
    def __init__(self):
        super(PreTrainedModel, self).__init__()
        """
        this function loads the resnet model and changes its input to fit
        black and white images.
        """
        self.modified_res_net = torchvision.models.resnet34(pretrained=True).\
            to('cuda')
        self.modified_res_net.conv1.weight =  \
            torch.nn.Parameter(self.modified_res_net.conv1.weight.sum(dim=1).
                               unsqueeze(1).data)

        self.proccesed_features = \
            torch.nn.Sequential(*list(self.modified_res_net.children())[0:8])

        self.long_skip_data = {}
        self.proccesed_features.eval()

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, bw_image):
        """
        passes the image through the model
        :param bw_image: a black and white image
        :return: the output of the first six layers
        """
        out = bw_image
        with torch.no_grad():
            for index in range(len(self.proccesed_features)):
                layer = self.proccesed_features[index]
                out = layer(out)
                if (type(layer) is nn.modules.container.Sequential or type(layer) is nn.ReLU) and index != 7:
                    self.long_skip_data[out.size()] = out

        return out


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
        self.decode_wights = nn.ModuleList()

        # initiate the weights
        self.init_weights_liniar_conv(structure)

        # create an optimizer for the network
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4,
                                          betas=(0, 0.999))

        self.scheduler = IterSetpScheduler(self.optimizer, step_size=1e5)

        # create the plots for debugging
        self.cost_plot = Plot("epochs", "cost", self.viz)
        self.accuracy_plot = Plot("epochs", "accuracy",  self.viz)

        self.pre_model = pre_model

        self.decoder_info = {}

        self.vgg_network = Vgg16()

        self.decode_index = 0

    def train_model(self, images, labels, rgb_image, discriminator=None, prep_layer=None, retain_graph=False):
        """
        trains the model, this function takes the training data and preforms
        the feed forward, back propagation and the optimisation process
        :return:
        """
        # feed forward through the network
        if prep_layer is None:
            middle_input = self.pre_model.forward(images).to(self.device)
            self.decoder_info = self.pre_model.long_skip_data
            prediction_layer = self.forward(
                Layer(middle_input, 0, net=self), decoder_flag=True).layer

        else:
            prediction_layer = prep_layer

        self.decoder_info.clear()
        rgb_image.to(self.device)
        # calculate the loss\
        fake_processed = torch.cat((images, prediction_layer), 1).to(self.device)
        decision_dis = discriminator.forward(
            Layer(fake_processed, 0, net=discriminator)).layer

        loss = - decision_dis.mean()

        c_loss = self.content_loss(prediction_layer, rgb_image.to(self.device), labels)

        comb_loss = loss + c_loss
        # backpropagate through the network

        self.optimizer.zero_grad()
        discriminator.optimizer.zero_grad()
        comb_loss.backward()

        self.optimizer.step()
        # save data for plots
        self.train_logger["loss"].append(comb_loss.item())

    def forward(self, layer, encoder_flag=False, decoder_flag=False):
        layer = self.forward_model(layer, encoder_flag, decoder_flag)
        self.decode_index = 0
        return layer

    def forward_model(self, layer, encoder_flag=False, decoder_flag=False):
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
            new_layer = self.create_activated_layer(layer, encoder_flag, decoder_flag)
            layer_1 = self.forward(new_layer, encoder_flag=encoder_flag, decoder_flag=decoder_flag)
            return layer_1

    def create_activated_layer(self, layer, encoder_flag, decoder_flag):
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

        if type(layer_params) is nn.ReLU and decoder_flag:
            try:
                calculated_layer = torch.cat(
                    (self.decode_wights[self.decode_index](self.decoder_info[
                        calculated_layer.size()]), calculated_layer), 1)
                self.decode_index += 1
            except KeyError:
                pass

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

    def content_loss(self, prediction_layer, expected_output, lab_image):
        prediction_layer.to(self.device)
        expected_output.to(self.device)
        x_pred = self.vgg_network.forward(prediction_layer)[:3]
        target_pred = self.vgg_network.forward(expected_output)[:3]
        base_loss = [self.l1_loss(prediction_layer, expected_output) / 100]
        wgts = [0.2, 0.7, 0.1]
        base_loss += [self.l1_loss(
            f.view(f.size(0), -1), t.view(f.size(0), -1))
                      * w for f, t, w in zip(x_pred, target_pred, wgts)]
        return sum(base_loss) * 100

    def init_weights_liniar_conv(self, sizes):
        for index in range(self.network_layers):
            if sizes[index][0] == 'lin':
                self.weights.append(nn.Linear(
                    sizes[index][1], sizes[index][2]).to(self.device))

            if sizes[index][0] == 'conv':
                self.weights.append(spectral_norm(nn.Conv2d(
                    sizes[index][1], sizes[index][2],
                    sizes[index][3], stride=sizes[index][4]).to(self.device)))
                """"
                self.weights[-1].weight.data.normal_(
                    0, np.sqrt(2. / sizes[index][1]), generator=
                    torch.cuda.manual_seed(100))
                """

            if sizes[index][0] == 'deconv':
                self.weights.append(spectral_norm(nn.ConvTranspose2d(
                    sizes[index][1], sizes[index][2] * 4,
                    sizes[index][3], stride=sizes[index][4]).to(self.device)))

                self.weights.append(nn.PixelShuffle(2))

                self.decode_wights.append(spectral_norm(nn.Conv2d(
                    sizes[index][2], sizes[index][2], kernel_size=1, stride=1))
                                          .to(self.device))

                """"
                self.weights[-1].weight.data.normal_(
                    0, np.sqrt(2. / sizes[index][1]),
                    generator=torch.cuda.manual_seed(100))
                """

            if sizes[index][0] == "decoder":
                self.weights.append(nn.Upsample(
                    scale_factor=sizes[index][1])).to(self.device)

            if sizes[index][0] == "batchnorm":
                self.weights.append(nn.BatchNorm2d(sizes[index][1]))\
                    .to(self.device)

            if sizes[index][0] == "relu":
                self.weights.append(nn.ReLU()).to(self.device)

            if sizes[index][0] == "tanh":
                self.weights.append(nn.Tanh()).to(self.device)

            if sizes[index][0] == "leaky":
                self.weights.append(nn.LeakyReLU(sizes[index][1]))\
                    .to(self.device)

            if sizes[index][0] == "selfAtt":
                self.weights.append(SelfAttention(sizes[index][1])).to(self.device)

            if sizes[index][0] == "dropout":
                self.weights.append(nn.Dropout2d(sizes[index][1]))

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
        self.eval()
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
                middle_input = self.pre_model.forward(images_gray).to(
                    self.device)
                self.decoder_info = self.pre_model.long_skip_data
                prediction_layer = self.forward(
                    Layer(middle_input, 0, net=self), decoder_flag=True).layer

                if display_data:
                    final = prediction_layer
                    if viz_win_res is None:
                        viz_win_res = self.viz.images(
                           final.cpu().numpy())
                    else:
                        self.viz.images(
                            final.cpu().numpy(), win=viz_win_res)

                self.test_logger["loss"].append(self.mse_loss(
                    prediction_layer, labels).item())

            self.test_logger["cost"].append(np.mean(self.test_logger["loss"]))
            self.test_logger["loss"] = []

    def eval_model(self):
        self.eval()
        im_gray = cv2.imread('test3.jpg', cv2.IMREAD_GRAYSCALE)
        im_gray = cv2.resize(im_gray, (256, 256))
        im_gray = im_gray.reshape(1, 1, 256, 256).astype('float32') / 256.0
        im_gray = torch.from_numpy(im_gray).to('cuda')
        with torch.no_grad():
            middle_input = self.pre_model.forward(im_gray).to(
                self.device)
            self.decoder_info = self.pre_model.long_skip_data
            colored = self.forward(Layer(middle_input, 0, net=self),
                                    decoder_flag=True).layer
            self.viz.image(colored.cpu()[0])


class Discriminator(NeuralNetwork):
    def __init__(
            self, viz_tool, pre_model, learnin_rate, structure,
            generator):
        super(Discriminator, self).__init__(
            viz_tool, pre_model, learnin_rate, structure)
        self.generator = generator
        self.cost_plot_discrim = Plot("epochs", "cost", self.viz)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-5,
                                          betas=(0, 0.999))

        self.scheduler = IterSetpScheduler(self.optimizer, step_size=1e5)

    def train_model(self, images, labels, rgb_image=None, discriminator=None, prep_layer=None, retain_graph=False):
            rgb_image = rgb_image.to(self.device)
            images = images.to(self.device)
            processed = torch.cat((images, rgb_image), 1).to(self.device)
            real_res = self.forward(Layer(processed.to(self.device), 0, net=self)).layer
            loss_real = nn.ReLU()(1.0 - real_res).mean()
            middle_input = self.generator.pre_model.forward(images).to(self.device)
            self.generator.decoder_info = self.generator.pre_model.long_skip_data
            fake_res = self.generator.forward(
                Layer(middle_input, 0, net=self.generator), decoder_flag=True).layer
            self.generator.decoder_info.clear()
            fake_processed = torch.cat((images, fake_res), 1).to(self.device)

            fake_res_dis = self.forward(Layer(fake_processed, 0, net=self)).layer
            loss_fake = nn.ReLU()(1.0 + fake_res_dis).mean()

            tot_loss = loss_real + loss_fake

            self.optimizer.zero_grad()
            self.generator.optimizer.zero_grad()
            tot_loss.backward()

            self.optimizer.step()

            self.train_logger["loss"].append(tot_loss.item())

            return fake_res.detach()


class CombinedTraining(object):
    def __init__(self, discriminator):
        self.discriminator = discriminator

    def super_train(self, epochs, train_loader, serial, images_per_epoch,
                    test_data=None):

        self.discriminator.train()
        self.discriminator.generator.train()
        for epoch in range(epochs):

            pbar = tqdm.tqdm(total=images_per_epoch)
            for i, (images, labels, rgb_image) in enumerate(train_loader):
                if i * 5 > images_per_epoch:
                    break

                images = images.to(self.discriminator.device)
                labels = labels.to(self.discriminator.device)

                # train the discriminator
                gen_out = self.discriminator.train_model(images, labels,
                                                         rgb_image=rgb_image)

                # train the generator

                self.discriminator.generator.train_model(
                    images, labels, rgb_image, discriminator=self.discriminator,
                    prep_layer=gen_out)

                self.discriminator.generator.train_model(
                    images, labels, rgb_image, discriminator=self.discriminator)

                pbar.update(5)

                if epoch < 3:
                    self.discriminator.scheduler.step()
                    self.discriminator.generator.scheduler.step()

                self.discriminator.scheduler.update()
                self.discriminator.generator.scheduler.update()
            pbar.close()
            tqdm.tqdm.write(
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
    model = None
    model.load_state_dict(torch.load(path))
    return model


def main(train_flag=True):
    # connect to the visdom server
    vis = visdom.Visdom()
    print("make sure visdom server is activated")

    # initialise data set
    train_loader, test_loader = DataParser.load_places_dataset(batch_size=5)
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
            ('deconv', 512, 256, 3, 1),
            ('relu', None),
            ('batchnorm', 512),
            ('deconv', 512, 128, 3, 1),
            ('relu', None),
            ('batchnorm', 256),
            ('selfAtt', 256),
            ('deconv', 256, 64, 3, 1),
            ('relu', None),
            ('batchnorm', 128),
            ('deconv', 128, 64, 3, 1),
            ('relu', None),
            ('batchnorm', 128),
            ('deconv', 128, 32, 3, 1),
            ('relu', None),
            ('batchnorm', 32),
            ('conv', 32, 3, 1, 1),
            ('tanh', None)])

    model2 = Discriminator(vis, None, 0.1, [('conv', 4, 128, 4, 2),
                                            ('leaky', 0.2),
                                            ('dropout', 0.2),
                                            ('conv', 128, 128, 3, 1),
                                            ('leaky', 0.2),
                                            ('dropout', 0.5),
                                            ('conv', 128, 256, 4, 2),
                                            ('leaky', 0.2),
                                            ('selfAtt', 256),
                                            ('dropout', 0.5),
                                            ('conv', 256, 512, 4, 2),
                                            ('leaky', 0.2),
                                            ('dropout', 0.5),
                                            ('conv', 512, 1024, 4, 2),
                                            ('leaky', 0.2),
                                            ('conv', 1024, 1, 4, 1)
                                            ], model)

    model.load_state_dict(torch.load("generator.ckpt"))
    model.test_model(test_loader, 600, True)
    trainer = CombinedTraining(model2)
    trainer.super_train(10, train_loader, '1', 5000,
                        test_loader)

    return model


if __name__ == '__main__':
    # set train flag to false to load pre trained model
    main(train_flag=True)
