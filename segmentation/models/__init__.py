# Assuming we have multiple models in this directory
from .unet import UNet
# from .model_1 import Model_2
# from .model_1 import Model_3

# Making a dictiaonary for convenience
existing_models = {
                    "unet": UNet
                    # "model_2": Model_2,
                    # "model_3": Model_3
}

def get_model(model_name, **kwargs):
    if model_name.lower() not in existing_models:
        raise ValueError(f"Model '{model_name}' han't been implemented. Choose one of: {list(existing_models.keys())} instead.")
    return existing_models[model_name.lower()](**kwargs)
