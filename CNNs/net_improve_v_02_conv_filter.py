import torch.nn as nn
import numpy as np

class ResizedConvFilterCNN(nn.Module):
    def __init__(self, kernel_size: int = 3, list_out_channels = [8, 16, 32], batch_norm = False, dropout_p = None):
        super().__init__()

        if kernel_size not in [3,5,7,9]:
            raise ValueError("Kernel size not feasible")
        
        if len(list_out_channels) > 5:
            raise ValueError("Net depth not feasible")

        padding = kernel_size // 2

        layers = []
        in_channels = 1
        input_side_length: int = 64

        for i, out_channels in enumerate(list_out_channels):
            block = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)]
            
            if batch_norm:
                block.append(nn.BatchNorm2d(out_channels))

            block.append(nn.ReLU())

            if dropout_p is not None:
                block.append(nn.Dropout2d(p = dropout_p))

            if i <= 3 and i < (len(list_out_channels) - 1):
                block.append(nn.MaxPool2d(2, 2))
                input_side_length //= 2
            
            layers.extend(block)

            in_channels = out_channels

        self.features = nn.Sequential(*layers)

        if dropout_p is not None:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_p),
                nn.Linear(input_side_length * input_side_length * list_out_channels[-1], 15)
            )
        else:
            self.classifier = nn.Linear(input_side_length * input_side_length * list_out_channels[-1], 15)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x