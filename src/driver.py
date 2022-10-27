import torch
import torchvision
import torchvision.datasets as datasets
from models import VGG16
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss


def train(args, loader):
    optimizer = torch.optim.Adam()
    loss_fn = CrossEntropyLoss()
    for epoch in epoch.args:
        print("Epoch", epoch)
        for images, label in loader:
            optimizer.zero_grad()
            images = images.to(args.device)
            label = label.to(args.device)

            out = model(images)
            pred = out
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

    return

def test():
    return



if __name__ == "__main__":
    # mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    # mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)
    

    # lfw_train = datasets.LFWPeople(root: './data', split: str = '10fold', image_set: str = 'funneled', transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False)
    t = transforms.Compose([transforms.Resize(224),  transforms.ToTensor()])
    lfw_train = datasets.LFWPeople(root= './data', split= 'Train', image_set='deepfunneled', transform=t , download= True)
    lfw_test = datasets.LFWPeople(root= './data', split= 'Test', image_set='deepfunneled', transform=t , download=True)
    print(len("Len data to train on"),lfw_train)

    train_loader = DataLoader(lfw_train, batch_size=16, shuffle=False)
    train_loader = DataLoader(lfw_test, batch_size=16, shuffle=False)

    model = VGG16()
    print(model)

    # image = Image.open('samples/AJ_Cook/AJ_Cook.jpg')
    # tensor_image = t(image)
    # print(tensor_image.shape)
    # inp = tensor_image.unsqueeze(dim=0)
    inp, y = next(iter(lfw_train))
    inp = inp.unsqueeze(dim=0)
    print(inp.shape)
    print('y')

    out = model(inp)
   
    for batch, batch_labels in train_loader:
        out = model(batch)
        print("Out")
        # print(out)
        print(out.shape)



    

