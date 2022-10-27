
import torch
import torch.nn as nn


class VGG_Model(nn.Module):
    
    def __init__(self):
        super().__init__()


        self.cnn_layers = nn.Sequential(
            # BLock 1
            nn.Conv2d(3,64,3,padding=1), 
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64,128,3,padding=1), 
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # Block 2
            nn.Conv2d(128,256,3,padding=1), 
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256,256,3,padding=1), 
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
        )

        # should be length of unwound channels * feature map dims
        db_size = 256 * 7 * 7
        # one layer to classify
        self.linear_layers = nn.Sequential(nn.Linear(db_size, 1024), nn.ELU(), nn.Linear(1024, 512), nn.ELU(), nn.Linear(512, 10), nn.ELU())
        self.fc = nn.Linear(10,3)


    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

