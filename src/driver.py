import torch
import torchvision
import torch.nn as nn
import torchvision.datasets as datasets
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



"""
Train Function

"""
# def train(args, model, loader):
#     #updates learning rates based on backward gradients of loss function
#     #initialize adam optimizer using specified learning rate
#     #   have to initialize after sending model to correct device
#     optimizer = Adam(model.parameters(), lr = args.lr) #momentum = 0.9

#     #initialize loss function
#     loss_fn = CrossEntropyLoss() #good for multiclass loss
#     running_loss = 0 #scope for return

#     #train images using specified epochs
#     for epoch in range(args.epochs):
#         running_loss = 0.
#         last_loss = 0.
#         print("----- Epoch " + str(epoch) + " -----")
#         for i, (images, label) in enumerate(loader):
#             optimizer.zero_grad() #zero learning gradient
#             images = images.to(args.device)
#             label = label.to(args.device)
#             print("label: ", label)
#             print(type(label))
#             print(label.shape)

#             #make predictions for batch
#             out = model(images) 
#             pred = out
#             print("prediction: ", pred)
            
#             #compute loss and gradient
#             loss = loss_fn(pred, label)
#             loss.backward()
            
#             #adjust learning rate
#             optimizer.step()

#             #gather data and report
#             running_loss += loss.item()
#             if i % 1000 == 999: #every 1000 losses, calculate loss per batch
#                 last_loss = running_loss / 1000 # loss per batch
#                 print('  batch {} loss: {}'.format(i + 1, last_loss))
#                 tb_x = epoch * len(loader) + i + 1
#                 print('Loss/train', last_loss)
#                 running_loss = 0

#     return running_loss #return running_loss for validation purposes

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

            # TODO - send out input image to tensorboard to assert that trasforms are in order
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

        # for batch_idx, (inputs, labels) in enumerate(val_loader):
        #     inputs = inputs.to(args.device)
        #     labels = labels.to(args.device)
        #     print("-----INPUTS-----")
        #     print(inputs)
        #     preds, _ = model(inputs)
        #     loss = loss_fn(preds, labels)

        #     val_loss += loss.item()
        #     other_val_loss += loss.item() 
        #     val_total_correct += preds.argmax(dim=1).eq(labels).sum().item()
        #     real_val_total_correct = preds.argmax(dim=1).eq(labels).sum().item()
        #     val_acc += real_val_total_correct / len(labels)
        #     num_batches_used = batch_idx + 1

        # val_loss = val_loss / num_batches_used
        # val_acc = val_acc / num_batches_used * 100
        # tb.add_scalar("ValLoss", val_loss, epoch)
        # tb.add_scalar("ValAccuracy", val_acc, epoch)




    tb.close()
    return

"""
Test Function

"""
def test(args, model, test_loader):
    print("Testing")
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (images, labels) in tqdm(test_loader):
            images = images.to(args.device)
            labels = labels.to(args.device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the {} test images: {} %'.format(10000, 100 * correct / total))  
    return



"""
Main Function

"""
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


    #download lfw deep funneled dataset if not downloaded
    # lfw_train = datasets.LFWPeople(root: './data', split: str = '10fold', image_set: str = 'funneled', transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False)
    t = transforms.Compose([transforms.Resize(args.resize),  transforms.ToTensor(), OccTransform(mask_size=args.resize)])
    lfw_train = datasets.LFWPeople(root= './data', split= 'Train', image_set='deepfunneled', transform=t,  download= True)
    lfw_test = datasets.LFWPeople(root= './data', split= 'Test', image_set='deepfunneled', transform=t , download=True)


    print("lfw_train: ", lfw_train)
    print("type lfw_train: ", type(lfw_train))

    lfw_train,  lfw_val = torch.utils.data.random_split(lfw_train, [.8, .2], generator=torch.Generator().manual_seed(42))
    print("lfw_train Length: ", lfw_train)
    lfw_train.transform = transforms.Compose([transforms.Resize(args.resize),  transforms.ToTensor(), OccTransform(mask_size=args.resize)])
    
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

    #create CNN based on vgg19 architecture
    
    print("========== Build Model==========")
    model = VGG19(in_channels=3, in_height=224, in_width=224, num_classes=5749, architecture=VGG_types["VGG19"])

   # model = VGG19(VGG_types["VGG19"])
    print("model: ", model)



    #if gpu is available, run on cuda enabled gpu
    if torch.cuda.is_available(): 
        model.cuda()

    print("========== Model Architecture ==========")
    print(model) #show model architecture

    print("========== Model Summary ==========")
    print(model, (1,3,224,224)) #model summary



    # inp, y = next(iter(lfw_train))
    # inp = inp.unsqueeze(dim=0)
    # print(inp.shape)
    # print('y', y)





    # image = Image.open("./samples/AJ_Cook/AJ_Cook.jpg")
    # image.show()
    # image = image.resize((224,224))

    # x = TF.to_tensor(image) #converts image into tensor array
    # #converts image (height, width, channel) into (batch_size, height, width, channel) 
    # x.unsqueeze_(0)            #(224, 224, 3) to (1, 3, 224, 224) 
    # x=x.to("cuda")             #convert to cuda
    # print(x.shape)             #testing shape 


    # pred = model(x)
    # pred_numpy = pred.detach().numpy()
    # pred_class = np.argmax(pred_numpy)
    # pred_class


    # torch.backends.cudnn.enabled==False #resolves - cuDNN error: CUDNN_STATUS_MAPPING_ERROR
    #RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED
    print("========== Train Model==========")
    train(args, model, train_loader, val_loader)

    print("========== Test Model==========")
    test(args, model, test_loader)

    # torch.save(model.state_dict(),  'model_weights.pth')

