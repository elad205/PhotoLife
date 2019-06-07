import torchvision
from torch import nn
import torch
from colorization.abc_net import NeuralNetwork, IterSetpScheduler
from colorization.layers import Layer, LayerTypes
import numpy as np
from colorization.loss import Loss
from colorization.DataParser import DataParser
import os
import sys


def no_grad(func):
    def do_no_grad(*args, **kwargs):
        with torch.no_grad():
            out = func(*args, **kwargs)
            return out
    return do_no_grad


class Vgg16(nn.Module):

    @no_grad
    def __init__(self, device):
        super(Vgg16, self).__init__()
        self.device = device
        self.vgg_pre_trained = torchvision.models.vgg16(
            pretrained=True).features.to(self.device)
        self.blocks = []
        self.vgg_pre_trained.eval()

        # extract the desired blocks from the entire network
        for i in range(4):
            self.blocks.append(nn.Sequential())

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

    @no_grad
    def forward(self, image):
        outputs = []
        out = image
        for block in self.blocks:
            outputs.append(block.forward(out))
            out = outputs[-1].to(self.device)

        return outputs


class ResNetEncoder(nn.Module):

    @no_grad
    def __init__(self, device):
        super(ResNetEncoder, self).__init__()
        """
        this function loads the resnet model and changes its input to fit
        black and white images.
        """
        self.device = device
        self.modified_res_net = torchvision.models.resnet34(
            pretrained=True)
        self.modified_res_net.eval()
        self.modified_res_net.conv1.weight =  \
            torch.nn.Parameter(
                self.modified_res_net.conv1.weight.sum(dim=1)
                    .unsqueeze(1).data)
        self.proccesed_features = \
            torch.nn.Sequential(*list(
                self.modified_res_net.children())[0:8]).to(self.device)

        self.long_skip_data = {}

        for param in self.parameters():
            param.requires_grad = False

    @no_grad
    def forward(self, bw_image):
        """
        passes the image through the model
        :param bw_image: a black and white image
        :return: the output of the first six layers
        """
        out = bw_image

        for index in range(len(self.proccesed_features)):
            layer = self.proccesed_features[index]
            out = layer(out)
            if (type(layer) is
                nn.modules.container.Sequential or type(
                        layer) is nn.ReLU) and index != 7:
                self.long_skip_data[out.size()] = out

        return out


class GeneratorDecoder(NeuralNetwork):
    def __init__(self, viz_tool, learning_rate, structure,
                 optimizer):
        super(GeneratorDecoder, self).__init__(
            viz_tool, learning_rate, structure, optimizer)

        self.decode_wights = torch.nn.ModuleList()
        self.pre_model = ResNetEncoder(self.device)
        self.decoder_info = {}
        self.vgg_network = Vgg16(self.device)
        self.decode_index = 0
        self.decoder_used = None

        self.dict_layers.update({"deconv": LayerTypes.deconv_layer,
                                "shuffle": LayerTypes.shuffle,
                                 "tanh": LayerTypes.tanh_layer,
                                 "selfAtt": LayerTypes.self_att_layer,
                                 "decoderBlock": LayerTypes.decoder_block
                                 })
        self.register_weights()
        self.optimizer = self.optimizer(self.parameters(),
                                        self.rate[0], self.rate[1])

        self.scheduler = IterSetpScheduler(self.optimizer, step_size=1e5)

    def register_weights(self):
        for layer in self.init_weights():
            if type(layer) is tuple:
                if type(layer[0]) is list:
                    for sub_layer in layer[0]:
                        self.weights.append(sub_layer.to(self.device))
                else:
                    self.weights.append(layer[0]).to(self.device)
                self.decode_wights.append(layer[1]).to(self.device)
            else:
                self.weights.append(layer).to(self.device)

    def forward(self, layer):
        layer = self.forward_model(layer)
        self.decode_index = 0
        return layer

    def create_activated_layer(self, layer):
        """
        creates the new layer in the network
        :param layer: layer objects
        :return: the next layer of the network a Layer object with the next
        index and if needed a flag for copy and concat the next layer
        """
        # create the layer of the model
        layer_params = self.weights[layer.index]

        # perform same padding on conv layers
        if type(layer_params) is nn.Conv2d or type(layer_params) is \
                nn.ConvTranspose2d:
            layer.calc_same_padding()

        # copy and concat before activation
        if type(layer_params) is nn.ReLU and layer.cat:
            try:
                self.concat_unet(layer)
            except KeyError:
                pass

        # get the next layer of the network
        calculated_layer = layer_params(layer.layer)

        # after sub pixel convolution copy and concat
        if type(layer_params) is nn.PixelShuffle:
            return Layer(layer_tensor=calculated_layer,
                         layer_number=layer.index + 1, net=layer.net, cat=True)

        # return the next layer
        return Layer(layer_tensor=calculated_layer,
                     layer_number=layer.index + 1, net=layer.net)

    def concat_unet(self, layer):
        """
        concat the current tensor of the network with the matching tesnor of
        the resent encoder
        :param layer: the current layer to be concat
        :return:
        """
        decoder_functions = self.decode_wights[self.decode_index]
        encoder_info = self.decoder_info[layer.layer.size()]
        layer.layer = torch.cat((
            layer.layer, decoder_functions(encoder_info)), dim=1)
        self.decode_index += 1

    @no_grad
    def test_model(
            self, test_data, images_per_epoch, batch_size, display_data=False):
        viz_win_images = None
        viz_win_res = None
        self.eval()
        for i, (images, labels) in enumerate(test_data):
            if (batch_size * i) >= images_per_epoch:
                break
            # display the images
            if display_data:
                disp = labels
                if viz_win_images is None and self.viz is not None:
                    viz_win_images = self.viz.images(labels)
                else:
                    if self.viz is not None:
                        self.viz.images(disp, win=viz_win_images)

            # parse the images and labels
            # feed forward through the network
            images = images.to(self.device)
            prediction_layer = self.feed_forward_generator(images)
            if display_data:
                final = prediction_layer
                if viz_win_res is None and self.viz is not None:
                    viz_win_res = self.viz.images(
                        final.cpu().numpy())
                else:
                    if self.viz is not None:
                        self.viz.images(
                            final.cpu().numpy(), win=viz_win_res)
            labels = labels.to(self.device)
            self.test_logger["loss"].append(Loss.mse_loss(
                prediction_layer, labels).item())

        self.test_logger["cost"].append(np.mean(self.test_logger["loss"]))
        self.test_logger["loss"] = []

    def train_model(self, images: torch.Tensor, labels: torch.Tensor,
                    extra_net):
        """
        trains the model, this function takes the training data and preforms
        the feed forward, back propagation and the optimisation process.
        In this function we implement the gans training scheme, we feed forward
        through the generator and then get what the discriminator thinks about
        the generated image and calculate the loss.
        We also use feature loss to try to replicate the image structure
        :param extra_net: an extra discriminator for the network or none
        :param images: the features for the network
        :param labels: the real images
        :return: None
        """
        # feed forward through the network
        if self.decoder_used is None:
            prediction_layer = self.feed_forward_generator(images)
        else:
            # use already generated image to save computing time
            prediction_layer = self.decoder_used

        self.decoder_info.clear()

        # calculate the loss
        if extra_net:
            decision_dis = extra_net.forward(
                Layer(prediction_layer, 0, net=extra_net)).layer

            # we try to fool the discriminator maximize the error, min(-error)
            loss = - decision_dis.mean()
        else:
            loss = 0

        # try to imitate the b&w image
        c_loss = Loss.content_loss(self, prediction_layer, labels)

        comb_loss = loss + c_loss

        self.optimizer.zero_grad()
        if extra_net:
            extra_net.optimizer.zero_grad()

        # back propagate through the network
        comb_loss.backward()

        # update the weights
        self.optimizer.step()
        self.decoder_used = None

        # save data for plots
        self.train_logger["loss"].append(comb_loss.item())

    def feed_forward_generator(self, images):
        middle_input = self.pre_model.forward(images).to(
            self.device)
        self.decoder_info = self.pre_model.long_skip_data
        prediction_layer = self.forward(
            Layer(middle_input, 0, net=self)).layer
        return prediction_layer

    @no_grad
    def eval_model(self, imgs, save_img=""):
        """
        this function just feeds forward through the model and saves the output
        :param imgs: the path ot the images to be processed
        :param save_img: the location to save them on
        :return:
        """
        self.eval()
        # convert the images to tensors
        arrays_images = DataParser.load_images(imgs)
        im_gray = torch.from_numpy(arrays_images).to(self.device)
        colored = self.feed_forward_generator(im_gray)
        if self.viz:
            self.viz.images(colored.cpu())
        if save_img != "":
            for img, name in zip(colored.cpu(), imgs):
                name = os.path.basename(name)
                try:
                    torchvision.utils.save_image(
                        img.cpu(), save_img + "\\" + name)
                    print(name, file=sys.stderr)
                except KeyError:
                    print("not saved")
                    print("0", file=sys.stderr)


class Discriminator(NeuralNetwork):
    def __init__(self,  viz_tool, learning_rate, structure, optimizer):
        super(Discriminator, self).__init__(
            viz_tool, learning_rate, structure, optimizer)

        self.dict_layers.update({"leaky": LayerTypes.leaky_relu_layer,
                                "dropout": LayerTypes.dropout_layer,
                                 "selfAtt": LayerTypes.self_att_layer,
                                 "convBlock": LayerTypes.discriminator_block
                                 })
        self.register_weights()
        self.optimizer = self.optimizer(self.parameters(),
                                        self.rate[0], self.rate[1])

        self.scheduler = IterSetpScheduler(self.optimizer, step_size=1e5)

    def train_model(self, images, labels, extra_net):
        """
        trains the model, this function takes the training data and preforms
        the feed forward, back propagation and the optimisation process.
        in this function we handle the discriminator training process.
        I used hinge loss as recommended for sagans. the goal in training
        the discriminator is classifying the images from the dataset and the
        images from the networks
        :param extra_net: an extra discriminator for the network or none
        :param images: the features for the network
        :param labels: the real images
        :return: None
        """

        # feed forward with real images
        real_res = self.forward(Layer(labels, 0, net=self)).layer

        # hinge loss for sagans, for real images we try to minimize the loss,
        # the loss output will be low if an image is real
        loss_real = Loss.hinge_loss(1.0 - real_res)

        # feed forward through the generator
        fake_res = extra_net.feed_forward_generator(images)

        extra_net.decoder_info.clear()
        # feed forward with the generated images
        fake_res_dis = self.forward(Layer(fake_res, 0, net=self)).layer

        # hinge loss for sagans, for fake images we try to minimize the loss,
        # thus making a the loss high if an image is fake
        loss_fake = Loss.hinge_loss(1.0 + fake_res_dis)
        tot_loss = loss_real + loss_fake

        self.optimizer.zero_grad()
        extra_net.optimizer.zero_grad()
        # calculate the gradients
        tot_loss.backward()

        # update the gradients
        self.optimizer.step()

        # log the loss
        self.train_logger["loss"].append(tot_loss.item())

        # save the result of the discriminator to save computing time
        extra_net.decoder_used = fake_res.detach()

    def forward(self, layer):
        return self.forward_model(layer)

    def test_model(
            self, test_data, images_per_epoch, batch_size, display_data=False):
        pass

    def register_weights(self):
        for layer in self.init_weights():
            self.weights.append(layer.to(self.device))


