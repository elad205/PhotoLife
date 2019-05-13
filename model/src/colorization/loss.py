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
        x_pred = gen.vgg_network.forward(prediction_layer)[:3]
        target_pred = gen.vgg_network.forward(expected_output)[:3]
        base_loss = [Loss.l1_loss(prediction_layer, expected_output)]
        wgts = [20, 70, 10]
        base_loss += [Loss.l1_loss(f, t) * w
                      for f, t, w in zip(x_pred, target_pred, wgts)]
        return sum(base_loss)
