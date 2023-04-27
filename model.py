import torch
import torch.nn as nn
import torch.nn.functional as F


class SRCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # first and last channels are 3 because we have RGB images
        # channels = [3, 64, 32, 3]
        # self.layers = nn.ModuleList([nn.Conv2d(channels[i], channels[i+1], kernel_size=3, padding=1) for i in range(len(channels)-1)])
        self.model = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size=3, padding=1), 
                        nn.ReLU(),
                        nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), 
                        nn.ReLU(),
                        nn.Conv2d(32, 3, kernel_size=3, padding=1)
                        )

        for l in self.model[:-1]:
            if isinstance(l, nn.Conv2d) or isinstance(l, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(l.weight.data, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(l.bias.data)
        # last layer
        # nn.init.normal_(self.model[-1].weight.data, 0.0, 0.001)
        # nn.init.normal_(self.model[-1].weight.data, 0.0, 0.01)
        # nn.init.zeros_(self.model[-1].bias.data)
    
    def forward(self, x):
        '''
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)
        '''
        return self.model(x)

class SRCNN2(nn.Module):
    def __init__(self):
        super().__init__()

        # first and last channels are 3 because we have RGB images
        self.model = nn.Sequential(
                        nn.Conv2d(3, 128, kernel_size=3, padding=1), 
                        nn.ReLU(),
                        nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), 
                        nn.ReLU(),
                        nn.Conv2d(64, 32, kernel_size=3, padding=1), 
                        nn.ReLU(),
                        nn.Conv2d(32, 3, kernel_size=3, padding=1)
                        )

        for l in self.model[:-1]:
            if isinstance(l, nn.Conv2d) or isinstance(l, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(l.weight.data, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(l.bias.data)
    
    def forward(self, x):
        return self.model(x)

class SRCNN3(nn.Module):
    def __init__(self):
        super().__init__()

        # first and last channels are 3 because we have RGB images
        self.model = nn.Sequential(
                        nn.Conv2d(3, 256, kernel_size=3, padding=1), 
                        nn.LeakyReLU(),
                        nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2), 
                        nn.BatchNorm2d(128),
                        nn.LeakyReLU(),
                        nn.Conv2d(128, 64, kernel_size=3, padding=1), 
                        nn.BatchNorm2d(64),
                        nn.LeakyReLU(),
                        nn.Conv2d(64, 32, kernel_size=3, padding=1), 
                        nn.BatchNorm2d(32),
                        nn.LeakyReLU(),
                        nn.Conv2d(32, 3, kernel_size=3, padding=1)
                        )

        for l in self.model[:-1]:
            if isinstance(l, nn.Conv2d) or isinstance(l, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(l.weight.data, mode='fan_out', nonlinearity='leaky_relu')
                nn.init.zeros_(l.bias.data)
    
    def forward(self, x):
        return self.model(x)

class SRCNN4(nn.Module):
    def __init__(self):
        super().__init__()

        # first and last channels are 3 because we have RGB images
        self.model = nn.Sequential(
                        nn.Conv2d(3, 128, kernel_size=3, padding=1), 
                        nn.LeakyReLU(),
                        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), 
                        nn.LeakyReLU(),
                        nn.Conv2d(64, 32, kernel_size=3, padding=1), 
                        nn.LeakyReLU(),
                        nn.Conv2d(32, 3, kernel_size=3, padding=1)
                        )

        for l in self.model[:-1]:
            if isinstance(l, nn.Conv2d) or isinstance(l, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(l.weight.data, mode='fan_out', nonlinearity='leaky_relu')
                nn.init.zeros_(l.bias.data)
    
    def forward(self, x):
        return self.model(x)