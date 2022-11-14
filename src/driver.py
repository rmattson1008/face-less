import torch
import torchvision
import torch.nn as nn
import torchvision.datasets as datasets
from models import VGG16
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import argparse
from data_utils import OccTransform
from tqdm import tqdm




def train(args, model, loader):
    #initialize adam optimizer using specified learning rate
    optimizer = Adam(model.parameters(), lr = args.lr)

    #initialize loss function
    loss_fn = CrossEntropyLoss()

    #train images using specified epochs
    for epoch in range(args.epochs):
        running_loss = 0.
        last_loss = 0.
        print("Epoch", epoch)
        for i, (images, label) in enumerate(loader):
            optimizer.zero_grad()
            images = images.to(args.device)
            label = label.to(args.device)
            print("label: ", label)
            print(type(label))
            print(label.shape)
            out = model(images)
            pred = out
            print("prediction: ", pred)
            
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

def test(args, model, test_loader):
    print("Testing")
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in tqdm(test_loader):
            images = images.to(args.device)
            labels = labels.to(args.device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the {} test images: {} %'.format(10000, 100 * correct / total))  
    return


if __name__ == "__main__":
    # mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    # mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

    #create argparse for command line input
    parser = argparse.ArgumentParser(prog = 'Face-less', 
                                    description = 'Facial classification using occluded data',
                                    epilog = 'Program uses default arguments if none are specified')
    parser.add_argument('--device', default='cpu', help='Default cpu if cuda is not available.')
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate for adam optimizer. Default 0.0001')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size. Default 32')
    parser.add_argument('--epochs', default=1, type=int, help='Number of epochs for CNN. Default 200')
    parser.add_argument('--resize', default=224, type=int, help='Size to resize input image to')
    args = parser.parse_args()

    #if gpu is available, run on cuda enabled gpu
    if torch.cuda.is_available(): 
        args.device = 'cuda'

    # lfw_train = datasets.LFWPeople(root: './data', split: str = '10fold', image_set: str = 'funneled', transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False)
    t = transforms.Compose([transforms.Resize(args.resize),  transforms.ToTensor(), OccTransform(mask_size=args.resize)])
    lfw_train = datasets.LFWPeople(root= './data', split= 'Train', image_set='deepfunneled', transform=t,  download= True)
    lfw_test = datasets.LFWPeople(root= './data', split= 'Test', image_set='deepfunneled', transform=t , download=True)
    print(len("Len data to train on"),lfw_train)


    #TODO - sample or randomize data
    train_loader = DataLoader(lfw_train, batch_size=16, shuffle=False)
    test_loader = DataLoader(lfw_test, batch_size=16, shuffle=False)

    #create CNN based on vgg19 architecture
    model = VGG16(num_classes=5749)

    #if gpu is available, run on cuda enabled gpu
    if torch.cuda.is_available(): 
        model.cuda()
    print(model) #show model architecture

    # inp, y = next(iter(lfw_train))
    # inp = inp.unsqueeze(dim=0)
    # print(inp.shape)
    # print('y', y)

    if (True): #create vgg19 model
        print("creating vgg model")








    train(args, model, train_loader)
    test(args, model, test_loader)

    

