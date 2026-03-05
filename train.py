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

# training loop
epochs= 10

for epoch in range(epochs):
    model.train()
    running_loss =0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        #forward pass
        output =model(images)
        loss = criterion(output, labels)

        #Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Accuracy calculation
        _, predicted = torch.max(output, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 *correct/ total
    print(f"Epoch[{epoch+1}/{epochs}] "
          f"Loss: {running_loss:.4f} "
          f"Accuracy: {accuracy:.2f}%")

print("Training finished!")

#save model - used in live run
torch.save(model.state_dict(), "emotion_cnn.pth")
print ("Model saved as emotion_cnn.pth")