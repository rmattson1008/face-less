import torch
import torchvision
import torchvision.datasets as datasets




if __name__ == "__main__":
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

    print(len("Len data to train on"), mnist_trainset)

