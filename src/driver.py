import torch
import torchvision
import torchvision.datasets as datasets
from models import VGG16
from torchvision import transforms
from PIL import Image



if __name__ == "__main__":
    # mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    # mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)
    # print(len("Len data to train on"), mnist_trainset)

    model = VGG16()
    print(model)

    image = Image.open('samples/AJ_Cook.jpg')
    t = transforms.Compose([transforms.Resize(224),  transforms.ToTensor()])
    tensor_image = t(image)
    print(tensor_image.shape)
    inp = tensor_image.unsqueeze(dim=0)
    print(inp.shape)

    out = model(inp)
    print("Out")
    print(out)
    print(out.shape)


    

