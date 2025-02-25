import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from utils.dataset import SegmentationDataset
import argparse
from models import get_model
import yaml

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Helper function to load the hyperparameters from a YAML file
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Parsing the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='unet', help='Model architecture to use')
parser.add_argument('--config', type=str, default='configs/unet.yaml', help='Path to the config file of the model being trained')
parser.add_argument('--checkpoint', type=str, default=None, help='Path to the checkpoint file to resume training or to fine tune the model')
args = parser.parse_args()

# Loading the hyperparameters from the YAML file
model_config = load_config(args.config)

in_channels = model_config['model']['in_channels']
out_channels = model_config['model']['out_channels']

num_epochs = model_config['training']['num_epochs']
learning_rate = model_config['training']['learning_rate']
batch_size = model_config['training']['batch_size']

# Creating the PyTorch DataLoader
train_dataset = SegmentationDataset(image_dir='data/images/train', mask_dir='data/masks/train') # Again, the paths are just placeholders as mentioned in the data directory README
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 

# Create DataLoader for validation set
val_dataset = SegmentationDataset(image_dir='data/images/val', mask_dir='data/masks/val') # Again, the paths are just placeholders as mentioned in the data directory README
val_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 

# Initializing the model, loss function, and the optimizer
model = get_model(args.model, in_channels=in_channels, out_channels=out_channels).to(device)
criterion = nn.BCELoss()  # Apparently this is the loss function to use for binary segmentation tasks. But I'm not sure if it's the best one (Use BCEWithLogitsLoss() if torch.forward() doesn't have a sigmoid layer)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Loading the checkpoint if provided
if args.checkpoint:
    model.load_state_dict(torch.load(args.checkpoint))

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
