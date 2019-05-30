from torch import nn
import torch


class Loss(object):
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

    @staticmethod
    def content_loss(gen, prediction_layer: torch.Tensor,
                     expected_output: torch.Tensor) -> torch.Tensor:
        """
        this is a feature loss. it tries to replicate the feature map of the
        pre trained network vgg 16. We do this by minimizing the l1 loss
        between the original image feature map  and the generator feature map
        :param gen: the generator network
        :param prediction_layer: the generated images of the generator
        :param expected_output: the original images
        :return: the calculated loss
        """
        # get the first 3 feature maps of the network
        x_pred = gen.vgg_network.forward(prediction_layer)[:3]

        # get the first 3 feature maps of the generator output
        target_pred = gen.vgg_network.forward(expected_output)[:3]

        # calculate the l1 loss between the maps
        base_loss = [Loss.l1_loss(prediction_layer, expected_output)]

        # the importance of each feature map
        weights = [20, 70, 10]
        # calc the loss
        base_loss += [Loss.l1_loss(f, t) * w
                      for f, t, w in zip(x_pred, target_pred, weights)]
        return sum(base_loss)

    @staticmethod
    def hinge_loss(delta):
        return nn.ReLU()(delta).mean()
