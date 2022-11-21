import torch
import torchvision
import torch.nn as nn
import torchvision.datasets as datasets
from torchvision import os
from models import VGG16 #custom vgg16 model
from vgg19 import VGG19
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import argparse
from data_utils import OccTransform
from tqdm import tqdm
import torchvision.transforms.functional as TF
import numpy as np
import random



"""
Setting seed for uses of random

    # shorturl.at/bKO38
"""
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def train(args, model, train_loader, val_loader):
    """ Train Function"""

    optimizer = Adam(model.parameters(), lr = args.lr)
    loss_fn = CrossEntropyLoss()

    total_correct = 0
    train_acc = 0

    comment = f' batch_size = {args.batch_size} lr = {args.lr} epochs = {args.epochs} {args.model_name}'
    tb = SummaryWriter(comment=comment)
    # log_dir = 'logs/tensorboard/'
    # tb = tf.summary.create_file_writer(log_dir)

    for epoch in range(args.epochs):
        train_loss = 0.0
        val_loss = 0.0
        train_acc = 0.0
        val_acc = 0.0
        num_batches_used = 0.0 

        print("Epoch", epoch)
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            images = images.to(args.device)
            labels = labels.to(args.device)

            # TODO - send out input image to tensorboard to assert that trasforms are in order
            logits = model(images)

            #what is last layer of vgg? 
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            train_loss += loss.item()

            train_acc += (preds.eq(labels).sum().item() / len(labels))
            
            num_batches_used = batch_idx + 1

        train_loss = (train_loss / num_batches_used)
        train_acc = train_acc / num_batches_used * 100
        tb.add_scalar("TrainLoss", train_loss, epoch)
        tb.add_scalar("TrainAccuracy", train_acc, epoch)

        print("validating...")
        for batch_idx, (inputs, labels) in enumerate(tqdm(val_loader)):
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            logits = model(inputs)
            loss = loss_fn(logits, labels)


            preds = logits.argmax(dim=1)
            # print(preds)
            # print(labels)

            val_loss += loss.item()
            # sum the accuracy for each batch, divide by number of batches later to get mean acc per epoch
            val_acc += (preds.eq(labels).sum().item() / len(labels))
            num_batches_used = batch_idx + 1

        val_loss = val_loss / num_batches_used
        val_acc = (val_acc / num_batches_used )* 100
        print("val acc ", val_acc)
        tb.add_scalar("ValLoss", val_loss, epoch)
        tb.add_scalar("ValAccuracy", val_acc, epoch)

    tb.close()
    return

def test(args, model, test_loader):
    print("Testing")
    with torch.no_grad():
        test_acc = 0
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader)):
            images = images.to(args.device)
            labels = labels.to(args.device)

            logits = model(images)
            preds = logits.argmax(dim=1)

            test_acc += (preds.eq(labels).sum().item() / len(labels))
            num_batches_used = batch_idx + 1

    test_acc = (test_acc / num_batches_used) * 100 
    print('Accuracy of the network on the test images: {} %'.format(test_acc)) 
    return

"""
Helper function for mean and sd for data normalization
"""
def batch_mean_and_sd(loader):
    
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in loader:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2,
                                  dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (
                      cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (
                            cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(
      snd_moment - fst_moment ** 2)        
    return mean,std
  


"""
Main Function

"""
if __name__ == "__main__":
    
# #create argparse for command line input
# args = argparse.ArgumentParser()
# args.add_argument('device')
# args.device = 'cpu'
# args.add_argument('lr')
# args.lr = 0.0001
# args.add_argument('batch_size')
# args.batch_size = 16
# args.add_argument('epochs')
# args.epochs = 1
# args.resize = 224
# print(args.epochs)


    #create argparse for command line input
    parser = argparse.ArgumentParser(prog = 'Face-less', 
                                    description = 'Facial classification using occluded data',
                                    epilog = 'Program uses default arguments if none are specified')
    parser.add_argument('--device', default='cpu', help='Default cpu if cuda is not available.')
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate for adam optimizer. Default 0.0001')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size. Default 32')
    parser.add_argument('--epochs', default=1, type=int, help='Number of epochs for CNN. Default 200')
    parser.add_argument('--resize', default=224, type=int, help='Size to resize input image to')
    parser.add_argument('--model_name', default="vgg", type=str, help='model nickname just to make life easier')
    # parser.add_argument('--resize', default=224, type=int, help='Size to resize input image to')
    args = parser.parse_args()

    #if gpu is available, run on cuda enabled gpu
    if torch.cuda.is_available(): 
        args.device = 'cuda'
        print("using cuda")


    #download lfw deep funneled dataset if not downloaded
    # lfw_train = datasets.LFWPeople(root: './data', split: str = '10fold', image_set: str = 'funneled', transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False)


    t1 = transforms.Compose([transforms.Resize(args.resize),  
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), #optional Normalization
        # OccTransform(True, mask_size=args.resize)
        ]
        ) #optional: boolean is for additional mask 

    t2 = transforms.Compose([transforms.Resize(args.resize),  
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) #optional: boolean is for additional mask 

    lfw_train = datasets.LFWPeople(root= './data', split= 'Train', image_set='deepfunneled', transform=t1,  download= True)
    lfw_test = datasets.LFWPeople(root= './data', split= 'Test', image_set='deepfunneled', transform=t2 , download=True)

    lfw_train, lfw_val = torch.utils.data.random_split(lfw_train, [.8, .2], generator=torch.Generator().manual_seed(42))

    # if issues with random split use this
    # train_size = int(.8 * len(lfw_train))
    # val_size = int(.2 * len(lfw_train))
    # assert train_size + val_size == len(lfw_train)
    # lfw_train,  lfw_val = torch.utils.data.random_split(lfw_train, [train_size, val_size], generator=torch.Generator().manual_seed(42))


    print("lfw_train: ", lfw_train)
    print("type lfw_train: ", type(lfw_train))
    print("lfw_train Length: ", lfw_train)


    train_loader = DataLoader(lfw_train, batch_size=16, shuffle=False)
    val_loader = DataLoader(lfw_val, batch_size=16, shuffle=False)
    test_loader = DataLoader(lfw_test, batch_size=16, shuffle=False)


    #VGG architectures for various implementaitons
    VGG_types = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [ 64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M", ],
    "VGG16": [ 64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M", ],
    "VGG19": [ 64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M", ],
    }


    print("========== Build Model==========")
    model = VGG19(in_channels=3, in_height=224, in_width=224, num_classes=5749, architecture=VGG_types["VGG19"])
    model.to(args.device)
    # print("model: ", model)



    # print("========== Model Architecture ==========")
    # print(model) #show model architecture


    # print("========== Model Summary ==========")
    # print(model, (1,3,224,224)) #model summary


    print("========== Train Model==========")
    train(args, model, train_loader, val_loader)

    print("========== Test Model==========")
    test(args, model, test_loader)

    # torch.save(model.state_dict(),  'model_weights.pth')



