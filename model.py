#   imports
import torch

import torch.nn as nn

class EmotionCNN(nn.Module):
    def __init__(self):
        super (EmotionCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d (2),  

            nn.Conv2d(32, 64, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2)  
        )

        self.fc_layers = nn.Sequential(
            nn.Linear (64 * 12 * 12, 128),
            nn.ReLU(),
            
            nn.Linear(128, 4)  #4 emotions - angry, happy,sad, nuetral
        )
