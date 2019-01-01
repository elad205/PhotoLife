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
        for feature, output_number in zip(input_layer, output_label):
            # convert the input layer into a pytorch tensor
            temp = np.array(feature).reshape(-1, 784)
            temp = temp.astype('float')
            torch_tensor = torch.from_numpy(temp)
            torch_tensor = torch_tensor.to(torch.device('cuda'))
            label = []
            output_number = np.array([output_number])
            output_number = output_number.astype('int64')
            output_number = torch.from_numpy(output_number)
            output_number = output_number.to(torch.device('cpu'))
            # create the first layer of the model
            first_layer = torch_tensor.mm(w1)
            # perform the activation function on every neuron [relu]
            first_layer = first_layer.clamp(min=0)
            # create the seconed layer of the model
            second_layer = first_layer.mm(w2)
            second_layer = second_layer.clamp(min=0)
            third_layer = second_layer.mm(w3)
            third_layer = third_layer.clamp(min=0)
            prediction_layer = third_layer.mm(w4)
            # perform softmax combined with nll loss
            loss_softmax = torch.nn.CrossEntropyLoss()
            prediction_layer = prediction_layer.to(torch.device('cpu'))
            loss = loss_softmax(prediction_layer, output_number)
            print(loss.item())

            # perform the backpropagation on the network
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
    print(labels[0])
    # print(np.asarray(images)[0].reshape(28,28))
    img = np.array(images[0]).astype('uint8').reshape(1, 28, 28)
    vis.image(img)
    img = np.array(images[0]).reshape(-1, 784)
    img = img.astype('float')
    print(img.dtype)
    n = NeuralNetwork(images, labels)


if __name__ == '__main__':
    main()
