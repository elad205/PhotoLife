import torch
from torch import nn
import torch.nn.functional as functional


class Layer:
    """
    this class represents a layer in the network, it contains the
    layer tensor and the place of the layer in the network
    """
    def __init__(self, layer_tensor: torch.Tensor, layer_number: int,
                 net, cat=False):
        self.layer = layer_tensor
        self.index = layer_number
        self.net = net
        self.cat = cat
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


class LayerTypes(object):

    @staticmethod
    def conv_layer(layer):
        return layer[-1](nn.Conv2d(
            layer[1], layer[2],
            layer[3], stride=layer[4]))

    @staticmethod
    def relu_layer(layer):
        return nn.ReLU()

    @staticmethod
    def batchnorm_layer(layer):
        return nn.BatchNorm2d(layer[2] * 2)

    @staticmethod
    def liniar_layer(layer):
        return nn.Linear(layer[1], layer[2])

    @staticmethod
    def deconv_layer(layer):
        c1 = layer[-1](nn.ConvTranspose2d(
                    layer[1], layer[2] * 4,
                    layer[3], stride=layer[4]))

        c2 = layer[-1](nn.Conv2d(
                   layer[2], layer[2], kernel_size=1, stride=1))

        return c1, c2

    @staticmethod
    def self_att_layer(layer):
        return SelfAttention(layer[1])

    @staticmethod
    def leaky_relu_layer(layer):
        if len(layer) > 3:
            index = -3
        else:
            index = 1
        return nn.LeakyReLU(layer[index])

    @staticmethod
    def dropout_layer(layer):
        return nn.Dropout2d(layer[-2])

    @staticmethod
    def tanh_layer(layer):
        return nn.Tanh()

    @staticmethod
    def shuffle(layer):
        return nn.PixelShuffle(2)

    @staticmethod
    def decoder_block(layer) -> tuple:
        block = list()
        deconv = LayerTypes.deconv_layer(layer)
        block.append(deconv[0])
        block.append(LayerTypes.shuffle(layer))
        block.append(LayerTypes.relu_layer(layer))
        block.append(LayerTypes.batchnorm_layer(layer))
        return block, deconv[1]

    @staticmethod
    def discriminator_block(layer, self_att=False) -> list:
        block = list()
        block.append(LayerTypes.conv_layer(layer))
        block.append(LayerTypes.leaky_relu_layer(layer))
        if self_att:
            block.append(LayerTypes.self_att_layer(layer))
        block.append(LayerTypes.dropout_layer(layer))

        return block


class SelfAttention(nn.Module):
    def __init__(self, in_channel: int):
        """
        taken from the fastai library
        """
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.query = self._spectral_init(
            nn.Conv1d(in_channel, in_channel // 8, 1))
        self.key = self._spectral_init(
            nn.Conv1d(in_channel, in_channel // 8, 1))
        self.value = self._spectral_init(
            nn.Conv1d(in_channel, in_channel, 1))
        self.gamma = nn.Parameter(torch.tensor(0.0)).to(self.device)

    @staticmethod
    def _spectral_init(module: nn.Module, gain: int = 1):
        nn.init.kaiming_uniform_(module.weight, gain)
        if module.bias is not None:
            module.bias.data.zero_()

        return nn.utils.spectral_norm(module)

    def forward(self, in_layer: torch.Tensor) -> torch.Tensor:
        in_layer = in_layer.to(self.device)
        size = in_layer.size()
        in_layer = in_layer.view(*size[:2], -1)
        f, g, h = self.query(in_layer), self.key(in_layer), self.value(in_layer)
        beta = functional.softmax(
            torch.bmm(f.permute(0, 2, 1).contiguous(), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + in_layer
        return o.view(*size).contiguous()
