import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from models.unet import UNet
from utils.dataset import SegmentationDataset
import argparse
from models import get_model

# Hyperparameters (Not sure if to leave this here or in the configs directory)
num_epochs = 100
learning_rate = 0.1
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# These hyperparameter values are just placeholders

# Creating the PyTorch DataLoader
train_dataset = SegmentationDataset(image_dir='data/images/train', mask_dir='data/masks/train') # Again, the paths are just placeholders as mentioned in the data directory README
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 

# Initializing the model, loss function, and the optimizer
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='unet', help='Model architecture to use')
args = parser.parse_args()

model = get_model(args.model, in_channels=3, out_channels=1).to(device)
criterion = nn.BCELoss()  # Apparently this is the loss function to use for binary segmentation tasks. But I'm not sure if it's the best one
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    
    model.train()
    running_loss = 0.0
    
    for batch in train_loader:

        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # TODO: Add extras like validation loop and more importantly checkpointing.
    
# Saving the model
torch.save(model.state_dict(), f"checkpoints/{args.model}/{args.model}_final.pth")
