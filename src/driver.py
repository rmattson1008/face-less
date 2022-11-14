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
from data_utils import OccTransform
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def train(args, model, train_loader, val_loader):
    optimizer = Adam(model.parameters(), lr = args.lr)
    loss_fn = CrossEntropyLoss()

    total_correct = 0
    train_acc = 0

    comment = f' batch_size = {args.batch_size} lr = {args.lr} epochs = {args.epochs} model_name'
    tb = SummaryWriter(comment=comment)
    for epoch in tqdm(range(args.epochs)):
        train_loss = 0.0
        val_loss = 0.0
        total_correct = 0.0
        val_total_correct = 0.0
        train_acc = 0.0
        val_acc = 0.0
        num_batches_used = 0.0 

        print("Epoch", epoch)
        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            images = images.to(args.device)
            labels = labels.to(args.device)
            # print(label)
            # print(type(label))
            # print(label.shape)
            out = model(images)
            pred = out
            # print(pred)
            
            loss = loss_fn(pred, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            total_correct += out.argmax(dim=1).eq(labels).sum().item()
            train_acc += total_correct / len(labels)
            num_batches_used = batch_idx + 1

        train_loss = train_loss / num_batches_used
        train_acc = train_acc / num_batches_used * 100
        tb.add_scalar("TrainLoss", train_loss, epoch)
        tb.add_scalar("TrainAccuracy", train_acc, epoch)

        for batch_idx, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            preds, _ = model(inputs)
            loss = loss_fn(preds, labels)

            val_loss += loss.item()
            other_val_loss += loss.item() 
            val_total_correct += preds.argmax(dim=1).eq(labels).sum().item()
            real_val_total_correct = preds.argmax(dim=1).eq(labels).sum().item()
            val_acc += real_val_total_correct / len(labels)
            num_batches_used = batch_idx + 1

        val_loss = val_loss / num_batches_used
        val_acc = val_acc / num_batches_used * 100
        tb.add_scalar("ValLoss", val_loss, epoch)
        tb.add_scalar("ValAccuracy", val_acc, epoch)




    tb.close()
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

    args = argparse.ArgumentParser()
    args.add_argument('device')
    args.device = 'cpu'
    args.add_argument('lr')
    args.lr = 0.0001
    args.add_argument('batch_size')
    args.batch_size = 16
    args.add_argument('epochs')
    args.epochs = 1
    args.resize=224
    print(args.epochs)
    

    # lfw_train = datasets.LFWPeople(root: './data', split: str = '10fold', image_set: str = 'funneled', transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False)
    t = transforms.Compose([transforms.Resize(args.resize),  transforms.ToTensor(), OccTransform(mask_size=args.resize)])
    lfw_train = datasets.LFWPeople(root= './data', split= 'Train', image_set='deepfunneled', transform=t,  download= True)
    lfw_test = datasets.LFWPeople(root= './data', split= 'Test', image_set='deepfunneled', transform=t , download=True)
    # split = len(lfw_train) * .8
    # end = len(lfw_train)
    lfw_train,  lfw_val = torch.utils.data.random_split(lfw_train, [.8, .2], generator=torch.Generator().manual_seed(42))
    print(len("Len data to train on"), lfw_train)

    #TODO - sample or randomize data
    
    train_loader = DataLoader(lfw_train, batch_size=16, shuffle=False)
    val_loader = DataLoader(lfw_val, batch_size=16, shuffle=False)
    test_loader = DataLoader(lfw_test, batch_size=16, shuffle=False)

    model = VGG16(num_classes=5749)
    print(model)

    # inp, y = next(iter(lfw_train))
    # inp = inp.unsqueeze(dim=0)
    # print(inp.shape)
    # print('y', y)

    train(args, model, train_loader, val_loader)
    test(args, model, test_loader)

    

