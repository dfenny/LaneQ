import os
import yaml
import time
import json
import argparse

import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.segmentation import MeanIoU
from tqdm.notebook import tqdm

from models import get_model
from utils.dataset import SegmentationDataset
from utils.preprocessing import get_img_transform


# global variable
DEVICE = "cpu"


# Helper function to load the hyperparameters from a YAML file
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def visualize_learning_curve(history, save_path, timestamp=""):

    plt.figure()
    plt.plot(history["train_loss"], label="train loss")
    plt.plot(history["val_loss"], label="val loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Training loss")

    fn = os.path.join(save_path, f"learning_curve_{timestamp}.png")
    plt.savefig(fn, bbox_inches='tight')
    plt.close()
    print(f"   Learning curve saved to {fn}")

    # save history for future use:
    fn = os.path.join(save_path, f"learning_history_{timestamp}.json")
    with open(fn, "w") as file:
        json.dump(history, file, indent=4)
    print(f"   Learning history saved to {fn}")


def generate_dataloader(image_dir, mask_dir, preprocess_config, batch_size, num_workers, shuffle=True, img_transform=None):

    # get required image transformation object
    resize_width, resize_height = preprocess_config["resize_width"], preprocess_config["resize_height"]
    if img_transform is None:
        image_transformations = get_img_transform(resize_height=resize_height, resize_width=resize_width)
    else:
        image_transformations = img_transform

    label_map = None
    if preprocess_config["RGB_mask"]:
        try:
            with open(preprocess_config["RGB_labelmap"], 'r') as json_file:
                label_map = json.load(json_file)
        except:
            print("Error loading labelmap")
            label_map = None

    # initialize dataset object
    dataset = SegmentationDataset(image_dir=image_dir, mask_dir=mask_dir,
                                  rgb_mask=preprocess_config["RGB_mask"], rgb_label_map=label_map,
                                  img_transform=image_transformations,
                                  mask_reshape=(resize_width, resize_height),
                                  one_hot_target=preprocess_config["one_hot_mask"],
                                  num_classes=preprocess_config["num_classes"])

    # generate dataloader
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader, len(dataset)


def train_loop(model, loss_fn, optimizer, train_loader, val_loader, num_epochs, save_path=".", checkpoint_freq=0):

    global DEVICE

    # ensure all necessary folders are available
    checkpoint_path = os.path.join(save_path, "checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)
    train_log_path = os.path.join(save_path, "train_log")
    os.makedirs(train_log_path, exist_ok=True)

    # initialize variables
    total_train_batch = len(train_loader)
    total_val_batch = len(val_loader)
    history = {"train_loss": [], "val_loss": []}

    # start training
    print("Training Started...")
    main_tic = time.time()
    for epoch in range(num_epochs):

        epoch_tic = time.time()
        # update weights using training data
        model.train()
        running_train_loss = 0
        for i, (batch_img, batch_mask) in enumerate(tqdm(train_loader, desc=f"Epoch: {epoch+1} train - ")):
            batch_img = batch_img.to(DEVICE)
            batch_mask = batch_mask.to(DEVICE)

            optimizer.zero_grad()               # set gradients to zero
            logits = model(batch_img)
            loss = loss_fn(logits, batch_mask)
            loss.backward()                     # calculate gradients
            optimizer.step()                    # take a step in optimization process
            running_train_loss += loss.item()

        epoch_train_loss = running_train_loss / total_train_batch   # average loss

        # check performance on validation set
        model.eval()  # Set the model to evaluation mode
        running_val_loss = 0
        with torch.no_grad():       # ensures that no gradients are computed

            for i, (batch_img, batch_mask) in enumerate(tqdm(val_loader, desc=f"Epoch: {epoch+1} val - ")):
                batch_img = batch_img.to(DEVICE)
                batch_mask = batch_mask.to(DEVICE)

                logits = model(batch_img)
                loss = loss_fn(logits, batch_mask)
                running_val_loss += loss.item()

        epoch_val_loss = running_val_loss / total_val_batch     # average loss

        # store to plot learning curve
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)

        epoch_toc = time.time()

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, "
              f"Epoch execution time: {round((epoch_toc-epoch_tic)/60, 2)} min")

        if checkpoint_freq > 0 and (epoch+1) % checkpoint_freq == 0:
            path = os.path.join(checkpoint_path, f"unet_checkpoint_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), path)

    # Saving the final model
    current_time = time.localtime()    # Get current time
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', current_time)   # Format as 'YYYY-MM-DD_HH-MM-SS'
    path = os.path.join(checkpoint_path, f"unet_final_{timestamp}.pth")
    torch.save(model.state_dict(), path)
    main_toc = time.time()

    # plot learning curve and save history
    visualize_learning_curve(history=history, save_path=train_log_path, timestamp=timestamp)

    print(f"Model saved at: {path}")
    print(f"Training Completed! Total time: {round((main_toc-main_tic)/60, 4)} min")


def cal_MeanIoU_score(model, data_loader, num_classes, per_class=True, foreground_th=0.5):

    global DEVICE
    meanIoU = MeanIoU(num_classes=num_classes, per_class=per_class, include_background=True, input_format='index')
    meanIoU = meanIoU.to(DEVICE)

    model.eval()
    with torch.no_grad():  # ensures that no gradients are computed
        for batch_img, batch_mask in data_loader:
            batch_img = batch_img.to(DEVICE)      # shape (batch size, num class, height, width)
            batch_mask = batch_mask.to(DEVICE)    # shape (batch size, num class, height, width)
            logits = model(batch_img)             # shape (batch size, num class, height, width)

            if logits.shape[1] == 1:              # if only 1 class  binary segmentation
                # Apply sigmoid to logits to get probabilities, then threshold to get binary class labels
                pred = torch.sigmoid(logits)                    # Sigmoid for binary classification     # (b, 1, h, w)
                pred_labels = (pred > foreground_th).float()    # Convert to 0 or 1 based on threshold  # (b, 1, h, w)
                pred_labels = pred_labels.squeeze(dim=1)        # (b, h, w)

                # accordingly update the batch_mask shape as well (done to ensure consistent input shape to meanIoU)
                batch_mask = batch_mask.squeeze(dim=1)        # (b, 1, h, w) -> (b, h, w)

            # multi-class segmentation
            else:
                prob = F.softmax(logits, dim=1)                 # convert to probs    # (b, c, h, w)
                pred_labels = torch.argmax(prob, dim=1)         # convert to labels   # (b, h, w)

                # accordingly update the batch_mask shape as well (done to ensure consistent input shape to meanIoU)
                batch_mask = torch.argmax(batch_mask, dim=1)    # one hot to label (b, c, h, w) -> (b, h, w)

            # Compute IoU for the current batch
            meanIoU.update(pred_labels.long(), batch_mask.long())

        # Calculate final estimate of meanIoU over entire dataset
        mIoU_score = meanIoU.compute()
        mIoU_score = mIoU_score.cpu().numpy()

    return mIoU_score


def main(model_name, config):

    global DEVICE

    # ensure all necessary folders are available
    os.makedirs(config["results_loc"], exist_ok=True)

    # get required config parameters
    model_config = config["model"]
    train_config = config["training"]
    dataset_config = config["dataset_loc"]
    preprocess_config = config["dataset_preprocessing"]

    if config["enable_cuda"]:
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using DEVICE: {DEVICE}")

    # generate train data loader
    train_loader, train_size = generate_dataloader(image_dir=dataset_config["train"]["img_dir"],
                                                   mask_dir=dataset_config["train"]["mask_dir"],
                                                   preprocess_config=preprocess_config,
                                                   batch_size=train_config["batch_size"],
                                                   num_workers=train_config["num_workers"])

    # generate validation data loader
    val_loader, val_size = generate_dataloader(image_dir=dataset_config["val"]["img_dir"],
                                               mask_dir=dataset_config["val"]["mask_dir"],
                                               preprocess_config=preprocess_config,
                                               batch_size=train_config["batch_size"],
                                               num_workers=train_config["num_workers"])

    print(f"Train Dataset loaded. #samples: {train_size}")
    print(f"Validation Dataset loaded. #samples: {val_size}")

    # Initializing the model, loss function, and the optimizer
    model = get_model(model_name, in_channels=model_config['in_channels'], out_channels=model_config['out_channels'])
    model = model.to(DEVICE)

    # Apparently this is the loss function to use for binary segmentation tasks.
    # But I'm not sure if it's the best one (Use BCEWithLogitsLoss() if torch.forward() doesn't have a sigmoid layer)
    # Using BCEWithLogitsLoss() as read in a blog that its numerically stable to use
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=train_config["learning_rate"])

    checkpoint_path = train_config["resume_checkpoint"]
    if checkpoint_path is not None:
        model.load_state_dict((torch.load(checkpoint_path, weights_only=True)))

    # train the model
    train_loop(model=model, loss_fn=criterion, optimizer=optimizer,
               train_loader=train_loader, val_loader=val_loader,
               num_epochs=train_config["num_epochs"], save_path=config["results_loc"],
               checkpoint_freq=train_config["save_checkpoint_freq"])

    # Evaluate model performance at end of training using mIoU
    print("Calculating Mean IoU Score (per class) ...")
    train_mIoU = cal_MeanIoU_score(model=model, data_loader=train_loader, num_classes=preprocess_config["num_classes"],
                                   foreground_th=config['inference']["foreground_threshold"])
    val_mIoU = cal_MeanIoU_score(model=model, data_loader=val_loader, num_classes=preprocess_config["num_classes"],
                                 foreground_th=config['inference']["foreground_threshold"])
    print("Train mIoU Score:", train_mIoU)
    print("Val mIoU Score:", val_mIoU)


if __name__ == '__main__':

    # Parsing the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='unet', help='Model architecture to use')
    parser.add_argument('--config', type=str, default='configs/unet.yaml',
                        help='Path to the config file of the model being trained')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to the checkpoint file to resume training or to fine tune the model')
    args = parser.parse_args()

    # Loading the hyperparameters from the YAML file
    config = load_config(args.config)

    # start training
    main(model_name=args.model, config=config)
