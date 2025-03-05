import torch
import torch.nn.functional as F

from models import get_model
from utils.preprocessing import apply_img_preprocessing


def load_saved_model(model_name, saved_weight_path,  **kwargs):
    model = get_model(model_name, **kwargs)       # initialize model
    model.load_state_dict((torch.load(saved_weight_path, weights_only=True)))   # load weights
    print("Saved weights loaded")
    return model


def pred_segmentation_mask(model, test_img, img_transform=None, add_batch_dim=False, pos_threshold=0.5,
                           device=torch.device("cpu")):

    # ensure model is on same device as test data
    model = model.to(device)

    # apply image transformations
    test_batch = apply_img_preprocessing(test_img, transform=img_transform)
    if add_batch_dim:
        test_batch = test_batch.unsqueeze(0)       # (b, 3, h, w)

    # Perform inference
    model.eval()
    with torch.no_grad():
        test_batch = test_batch.to(device)
        logits = model(test_batch)            # (b, c, h, w)

        if logits.shape[1] == 1:              # if only 1 class  binary segmentation
            # Apply sigmoid to logits to get probabilities, then threshold to get binary class labels
            pred = torch.sigmoid(logits)  # Sigmoid for binary classification      # (b, c, h, w)
            pred_labels = (pred > pos_threshold).float()  # Convert to 0 or 1 based on threshold    # (b, 1, h, w)
            pred_labels = pred_labels.squeeze(dim=1)  # (b, h, w)

        # multi-class segmentation
        else:
            prob = F.softmax(logits, dim=1)  # convert to probs   (b, c, h, w)
            pred_labels = torch.argmax(prob, dim=1)  # convert to labels  (b, h, w)

    # bring pred on cpu
    pred_labels = pred_labels.cpu().numpy()
    return pred_labels
