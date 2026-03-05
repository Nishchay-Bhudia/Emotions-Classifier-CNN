#  imports
import torch
import torch.nn as nn          
import torch.optim as optim

from dataset import get_dataloaders
from model import EmotionCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Load  data
train_loader, test_loader =get_dataloaders("data", batch_size=64)

#Initialize model
model =EmotionCNN().to(device)

#   loss + Optimizer
criterion =nn.CrossEntropyLoss()
optimizer= optim.Adam(model.parameters(), lr=0.001)
