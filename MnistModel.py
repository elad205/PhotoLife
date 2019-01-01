import mnist
import numpy as np
import torch
import visdom
import VisdomServer
import math
import torchvision


class NeuralNetwork:
    def __init__(self, input_layer, output_label):

        # using the nvidia cuda api in order to perform calculations on a GPU
        device = torch.device('cuda')
        self.cost = (0, 0)
        self.rate = 1e-6

        # initialize the weights - 4 layers total
        w1 = torch.randn(784, 2500, device=device, dtype=torch.double, requires_grad=True)
        w2 = torch.randn(2500, 1000, device=device, dtype=torch.double, requires_grad=True)
        w3 = torch.randn(1000, 500, device=device, dtype=torch.double, requires_grad=True)
        w4 = torch.randn(500, 10,  device=device, dtype=torch.double, requires_grad=True)

        # convert the input layer into a pytorch tensor
        torch_tensor = torch.from_numpy(input_layer)
        torch_tensor = torch_tensor.to(torch.device('cuda'))

        # create the first layer of the model
        first_layer = torch_tensor.mm(w1)
        print(first_layer.size)
        # perform the activation function on every neuron [relu]
        first_layer = first_layer.clamp(min=0)
        # create the seconed layer of the model
        second_layer = first_layer.mm(w2)
        second_layer = second_layer.clamp(min=0)
        third_layer = second_layer.mm(w3)
        third_layer = third_layer.clamp(min=0)
        prediction_layer = third_layer.mm(w4)
        soft_max = torch.nn.LogSoftmax()
        soft_max_layer = soft_max(prediction_layer)
        # calculate the loss
        # loss = (prediction_layer - output_label).pow(2).sum()lk
        loss = (-soft_max_layer).sum()
        """
        nu = soft_max_layer.to(torch.device('cpu')).detach().numpy()
        print(type(nu))
        loss = (-np.log(nu)).sum()
        loss = torch.from_numpy(loss)
        loss = loss.to(torch.device('cuda'))
        print(loss)
        """
        torch.autograd.backward(loss)

        with torch.no_grad():
            w1 -= self.rate * w1.grad
            w2 -= self.rate * w2.grad
            w3 -= self.rate * w3.grad
            w4 -= self.rate * w4.grad

            # Manually zero the gradients after running the backward pass
            w1.grad.zero_()
            w2.grad.zero_()
            w3.grad.zero_()
            w4.grad.zero_()

def main():
    ser = VisdomServer.Server()
    vis = visdom.Visdom()
    vis.text('Hello, world!')
    data = mnist.MNIST(r"C:\Users\eladc\PycharmProjects\PyTorchEnv\MNIST")
    images, labels = data.load_training()
    # print(np.asarray(images)[0].reshape(28,28))
    img = np.array(images[0]).astype('uint8').reshape(1,28, 28)
    vis.image(img)
    img = np.array(images[0]).reshape(-1, 784)
    img = img.astype('float')
    print(img.dtype)
    n = NeuralNetwork(img, labels[0])


if __name__ == '__main__':
    main()
