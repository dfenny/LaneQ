# Assuming we have multiple models in this directory
from .unet import UNet
from .cnn import CNN
from .cnn_sppf import CNN_SPPF

# Making a dictionary for convenience
_existing_models = {
                    "unet": UNet,
                    "cnn": CNN,
                    "cnn_sppf": CNN_SPPF,
}


def get_model(model_name, **kwargs):

    if model_name.lower() not in _existing_models:
        raise ValueError(f"Model '{model_name}' hasn't been implemented. Choose one of: {list(_existing_models.keys())} instead.")

    return _existing_models[model_name.lower()](**kwargs)
