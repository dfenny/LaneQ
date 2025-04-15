import os
import argparse
import time
import pandas as pd
from tqdm import tqdm

import torch
import torch.optim as optim
from torch import nn

from .architectures import get_model
from .dataloader import generate_dataloader
from .focal_loss import FocalLoss
from .evaluation_metrics import cal_MeanIoU_score, cal_classification_metrics, cal_regression_metrics
from .utils.common import load_config, visualize_learning_curve, visualize_confusion_matrix

# global variable
DEVICE = "cpu"

_loss_fn_map = {
    "BCE": nn.BCEWithLogitsLoss,
    "CrossEntropy": nn.CrossEntropyLoss,
    "MSE": nn.MSELoss,
    "FocalLoss": FocalLoss,
}


def get_loss(loss_fn_name, **kwargs):
    if loss_fn_name not in _loss_fn_map:
        raise ValueError(f"Loss function '{loss_fn_name}' not supported.")
    return _loss_fn_map[loss_fn_name](**kwargs)


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
        model.train()                    # update weights using training data
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


def main(model_name, config):

    global DEVICE

    # ensure all necessary folders are available
    os.makedirs(config["results_loc"], exist_ok=True)

    # get required config parameters
    model_config = config["model"]
    dataset_config = config["dataset_loc"]
    dataloader_config = config["data_loader"]
    preprocess_config = config.get("dataset_preprocessing", None)
    train_config = config["training"]

    if config["enable_cuda"]:
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using DEVICE: {DEVICE}")

    # generate train data loader
    train_loader, train_size = generate_dataloader(dataset_type=dataset_config["dataset_type"],
                                                   data_loc=dataset_config["train"],
                                                   dataloader_config=dataloader_config,
                                                   preprocess_config=preprocess_config)

    # generate validation data loader
    val_loader, val_size = generate_dataloader(dataset_type=dataset_config["dataset_type"],
                                               data_loc=dataset_config["val"],
                                               dataloader_config=dataloader_config,
                                               preprocess_config=preprocess_config)

    print(f"Train Dataset loaded. #samples: {train_size}")
    print(f"Validation Dataset loaded. #samples: {val_size}")

    # Initializing the model, loss function, and the optimizer
    model = get_model(model_name, in_channels=model_config['in_channels'], out_channels=model_config['out_channels'])
    model = model.to(DEVICE)

    criterion = get_loss(train_config["loss_fn_name"])
    optimizer = optim.Adam(model.parameters(), lr=train_config["learning_rate"])

    checkpoint_path = train_config["resume_checkpoint"]
    if checkpoint_path is not None:
        model.load_state_dict((torch.load(checkpoint_path, weights_only=True)))
        print("Model checkpoint loaded.")

    # train the model
    train_loop(model=model, loss_fn=criterion, optimizer=optimizer,
               train_loader=train_loader, val_loader=val_loader,
               num_epochs=train_config["num_epochs"], save_path=config["results_loc"],
               checkpoint_freq=train_config["save_checkpoint_freq"])

    # Evaluate model performance at end of training using mIoU
    print("Calculating evaluation metrics ...")
    if dataset_config["dataset_type"] == "segmentation":

        train_mIoU = cal_MeanIoU_score(model=model, data_loader=train_loader,
                                       num_classes=preprocess_config.get("num_classes", 2), device=DEVICE)
        val_mIoU = cal_MeanIoU_score(model=model, data_loader=val_loader,
                                     num_classes=preprocess_config.get("num_classes", 2), device=DEVICE)
        print("Train mIoU Score:", train_mIoU)
        print("Val mIoU Score:", val_mIoU)

    elif dataset_config["dataset_type"] == "classification":

        train_metrics, train_cm = cal_classification_metrics(model, train_loader, device=DEVICE)
        val_metrics, val_cm = cal_classification_metrics(model, val_loader, device=DEVICE)
        stats = pd.DataFrame([train_metrics, val_metrics], index=["train", "val"]).T
        print("Model metrics:\n", stats)
        visualize_confusion_matrix(train_cm=train_cm, val_cm=val_cm, save_path=config["results_loc"])

    elif dataset_config["dataset_type"] == "regression":

        train_losses = cal_regression_metrics(model, train_loader)
        val_losses = cal_regression_metrics(model, val_loader)
        print(f"Train Loss: {train_losses}")
        print(f"Validation Loss: {val_losses}")


if __name__ == '__main__':

    # Parsing the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='unet', help='Model architecture to use')
    parser.add_argument('--config', type=str, default='configs/unet.yaml',
                        help='Path to the config file of the model being trained')
    args = parser.parse_args()

    # Loading the hyperparameters from the YAML file
    config = load_config(args.config)

    # start training
    main(model_name=args.model, config=config)

