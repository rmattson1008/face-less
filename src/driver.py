import torch
import torchvision
import torchvision.datasets as datasets
from models import VGG_Model




if __name__ == "__main__":
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)
    print(len("Len data to train on"), mnist_trainset)

    model = VGG_Model()
    print(model)

