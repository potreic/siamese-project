import torch
import torch.nn as nn
import torch.optim as optim

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.conv_sequential = nn.Sequential(
            # Layer 1: 64 filters of 10x10
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=10, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer 2: 128 filters of 7x7
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer 3: 128 filters of 4x4
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer 4: 256 filters of 4x4
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=1),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Sequential(
            nn.Linear(9216, 4066), # output conv 256x6x6 = 9216
            nn.Sigmoid()
        )

        self.out = nn.Sequential(
            nn.Linear(4096, 1),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=1e-2)
                if m.bias is not None:
                    nn.init.normal_(m.bias, mean=0.5, std=1e-2)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=2e-1)
                nn.init.normal_(m.bias, mean=0.5, std=1e-2)

    def forward_one(self, x):
        x = self.conv_sequential(x)
        x = x.view(x.size()[0], -1) 
        x = self.fc(x)
        return x             

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        
        l1_distance = torch.abs(out1-out2)

        output = self.out(l1_distance)
        return output
    
# model = SiameseNetwork()

# batch_size = 128
# criterion = nn.BCELoss()
# initial_lr = 0.01
# l2_penalty = 0.05

# optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.5, weight_decay=l2_penalty)
# lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
