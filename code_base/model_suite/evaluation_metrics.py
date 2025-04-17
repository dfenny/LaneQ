from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchmetrics.segmentation import MeanIoU

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix



def cal_MeanIoU_score(model, data_loader, num_classes, per_class=True, foreground_th=0.5, device="cpu"):
    """
    Compute the mean Intersection-over-Union (IoU) score for a segmentation model.

    Parameters
    ----------
    model : torch.nn.Module
        Trained segmentation model.
    data_loader : torch.utils.data.DataLoader
        DataLoader yielding (images, masks) pairs where masks are one-hot or single-channel tensors.
    num_classes : int
        Number of classes (including background) for IoU computation.
    per_class : bool, default=True
        If True, returns IoU for each class; otherwise returns overall mean IoU.
    foreground_th : float, default=0.5
        Threshold for converting sigmoid outputs to binary masks in binary segmentation.
    device : str or torch.device, default="cpu"
        Device on which to perform computations.

    Returns
    -------
    numpy.ndarray or float
        If `per_class` is True, an array of shape (num_classes,) with IoU per class;
        otherwise a single float scalar with the overall mean IoU.
    """
    meanIoU = MeanIoU(num_classes=num_classes, per_class=per_class, include_background=True, input_format='index')
    meanIoU = meanIoU.to(device)

    model.eval()
    with torch.no_grad():  # ensures that no gradients are computed
        for batch_img, batch_mask in tqdm(data_loader):
            batch_img = batch_img.to(device)      # shape (batch size, num class, height, width)
            batch_mask = batch_mask.to(device)    # shape (batch size, num class, height, width)
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


def cal_regression_metrics(model, data_loader, device="cpu"):
    """
    Compute common regression metrics (MSE, MAE, R²) for a regression model.

    Parameters
    ----------
    model : torch.nn.Module
        Trained regression model.
    data_loader : torch.utils.data.DataLoader
        DataLoader yielding (images, target_value) pairs.
    device : str or torch.device, default="cpu"
        Device on which to perform inference.

    Returns
    -------
    dict
        Dictionary with keys:
          - "MSE": Mean Squared Error (float, rounded to 4 decimals)
          - "MAE": Mean Absolute Error (float, rounded to 4 decimals)
          - "R2" : R² score (float, rounded to 4 decimals)
    """
    model.to(device).eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch_img, batch_target in tqdm(data_loader):
            batch_img = batch_img.to(torch.float32).to(device)
            batch_target = batch_target.to(torch.float32).to(device)

            preds = model(batch_img)
            y_true.extend(batch_target.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {"MSE": round(mse, 4), "MAE": round(mae, 4), "R2": round(r2, 4)}


def cal_classification_metrics(model, data_loader, device="cpu"):
    """
    Compute classification performance metrics and confusion matrix for a classifier.

    Parameters
    ----------
    model : torch.nn.Module
        Trained classification model.
    data_loader : torch.utils.data.DataLoader
        DataLoader yielding (images, one-hot-labels) pairs.
    device : str or torch.device, default="cpu"
        Device on which to perform inference.

    Returns
    -------
    metrics : dict
        Dictionary with rounded (4 decimals) values for:
          - 'accuracy'
          - 'precision' (weighted)
          - 'recall'    (weighted)
          - 'f1'        (weighted)
    cm : ndarray
        Normalized confusion matrix of shape (n_classes, n_classes).
    """
    model.to(device).eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch_img, batch_target in tqdm(data_loader):
            batch_img = batch_img.to(torch.float32).to(device)
            batch_target = batch_target.to(torch.long).to(device)  # Ensure targets are LongTensor for classification

            logits = model(batch_img)  # Forward pass through the model
            probs = F.softmax(logits, dim=1)  # Apply softmax to get probabilities (one-hot encoded format)
            preds = torch.argmax(probs, dim=1)  # Get predicted class (argmax)

            batch_target = torch.argmax(batch_target, dim=1)  # Convert one-hot target to class indices
            y_true.extend(batch_target.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred, normalize='all')
    metrics = {
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4)
    }

    return metrics, cm



