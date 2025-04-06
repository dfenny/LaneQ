import torch
import torch.nn.functional as F

# Import these if running /regression/Regression_train_inference.ipynb
# from models import get_model
# from utils.preprocessing import apply_img_preprocessing

# Import these if running /inference.py
from .models import get_model
from .utils.preprocessing import apply_img_preprocessing


def load_saved_model(model_name, saved_weight_path,  **kwargs):
    model = get_model(model_name, **kwargs)       # initialize model
    model.load_state_dict((torch.load(saved_weight_path, weights_only=True)))   # load weights
    # print("Saved weights loaded")
    return model


def pred_degradation_value(model, test_img, img_transform=None, add_batch_dim=False, pos_threshold=0.5,
                           device=torch.device("cpu")):

    # ensure model is on same device as test data
    model = model.to(device)

    # apply image transformations
    test_batch = apply_img_preprocessing(test_img, transform=img_transform)
    if add_batch_dim:
        test_batch = test_batch.unsqueeze(0)       # (b, 3, h, w)

    model.eval()
    with torch.no_grad():
        test_batch = test_batch.to(device)
        output = model(test_batch)

        pred_value = output.squeeze().cpu().item()

    return pred_value
