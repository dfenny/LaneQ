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
    """
    Retrieve and initialize a model by its name.
    Parameters
    ----------
    model_name : str
        The name of the model to retrieve. Must match one of the keys in `_existing_models`.
    **kwargs : dict, optional
        Additional keyword arguments to pass to the model's initialization function.
    Returns
    -------
    object
        An instance of the requested model, initialized with the provided arguments.
    """

    if model_name.lower() not in _existing_models:
        raise ValueError(f"Model '{model_name}' hasn't been implemented. Choose one of: {list(_existing_models.keys())} instead.")

    return _existing_models[model_name.lower()](**kwargs)
