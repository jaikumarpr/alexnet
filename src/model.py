import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, dropout, num_classes):
        super(AlexNet, self).__init__()
        
        self.conv_layer1 = nn.Sequential(
            # input size of 227 x 227 with 96 kernels of filter 11 with stride 4
            nn.Conv2d(3, 96, 11, 4), 
            # ->  55 x 55 x 96
            
            nn.BatchNorm2d(96),
            nn.ReLU(),
            
            # max pooling of kernel 3 with stride 2 for overlapping pooling
            nn.MaxPool2d(3, 2),
            # -> 27 x 27 x 96
        )

        self.conv_layer2 = nn.Sequential(
            # 256 kernels with size 5
            # add padding to maintain output 27
            nn.Conv2d(96, 256, 5, padding=2),
            # -> 27 x 27 x 256
            
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # max pooling of kernel 3 with stride 2 for overlapping pooling 
            nn.MaxPool2d(3, 2)
            # -> 13 x 13 x 256
        )
        
        self.conv_layer3 = nn.Sequential(
            # 384 kernels of filter size 3
            # add padding to keep the out 13
            nn.Conv2d(256, 384, 3, padding=1),
            # 13 x 13 x 384
            nn.ReLU(),
        )

        self.conv_layer4 = nn.Sequential(
            # 384 kernels of filter 3
            # add padding to keep the out 13
            nn.Conv2d(384, 384, 3, padding=1),
            # -> 13 x 13 x 384 

            nn.ReLU(),
        )

        self.conv_layer5 = nn.Sequential(
            # 256 kernels of filter 3
            nn.Conv2d(384, 256, 3, padding=1),
            # 13 x 13 x 256
            nn.ReLU(),
            
            # max pooling of kernel 3 with stride 2 for overlapping pooling 
            nn.MaxPool2d(3, 2),
            # -> 6 x 6 x 256
        )
        
        self.fully_connected = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(6 * 6 * 256, 4096),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = self.conv_layer5(x)
        x = self.fully_connected(x)
        
        return x