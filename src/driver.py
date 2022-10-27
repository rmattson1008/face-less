import torch
import torchvision
import torchvision.datasets as datasets
from models import VGG16
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import argparse


def train(args, model, loader):
    optimizer = Adam(model.parameters(), lr = args.lr)
    loss_fn = CrossEntropyLoss()
    for epoch in range(args.epochs):
        running_loss = 0.
        last_loss = 0.
        print("Epoch", epoch)
        for i, (images, label) in enumerate(loader):
            optimizer.zero_grad()
            images = images.to(args.device)
            label = label.to(args.device)
            print(label)
            print(type(label))
            print(label.shape)
            out = model(images)
            pred = out
            print(pred)
            
            loss = loss_fn(pred, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch * len(loader) + i + 1
                print('Loss/train', last_loss)
                running_loss = 0

    return

def test():
    return



if __name__ == "__main__":
    # mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    # mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

    args = argparse.ArgumentParser()
    args.add_argument('device')
    args.device = 'cpu'
    args.add_argument('lr')
    args.lr = 0.0001
    args.add_argument('batch_size')
    args.batch_size = 16
    args.add_argument('epochs')
    args.epochs = 10
    print(args.epochs)
    

    # lfw_train = datasets.LFWPeople(root: './data', split: str = '10fold', image_set: str = 'funneled', transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False)
    t = transforms.Compose([transforms.Resize(224),  transforms.ToTensor()])
    lfw_train = datasets.LFWPeople(root= './data', split= 'Train', image_set='deepfunneled', transform=t,  download= True)
    lfw_test = datasets.LFWPeople(root= './data', split= 'Test', image_set='deepfunneled', transform=t , download=True)
    print(len("Len data to train on"),lfw_train)

    #TODO - sample or randomize data
    train_loader = DataLoader(lfw_train, batch_size=16, shuffle=False)
    train_loader = DataLoader(lfw_test, batch_size=16, shuffle=False)

    model = VGG16(num_classes=5749)
    print(model)

    # image = Image.open('samples/AJ_Cook/AJ_Cook.jpg')
    # tensor_image = t(image)
    # print(tensor_image.shape)
    # inp = tensor_image.unsqueeze(dim=0)
    inp, y = next(iter(lfw_train))
    inp = inp.unsqueeze(dim=0)
    print(inp.shape)
    print('y')

    train(args, model, train_loader)


    out = model(inp)
   
    for batch, batch_labels in train_loader:
        out = model(batch)
        print("Out")
        # print(out)
        print(out.shape)



    

