# dataset.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define image transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  #convert to 1 channel
    transforms.Resize((48, 48)),                  # resize to 48x48
    transforms.ToTensor(),                        #convert to tensor
    transforms.Normalize((0.5,), (0.5,))         #normalize to [-1,1]
])

def get_dataloaders(data_dir, batch_size=64):
   
    train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
    test_dataset = datasets.ImageFolder(root=f"{data_dir}/test", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader