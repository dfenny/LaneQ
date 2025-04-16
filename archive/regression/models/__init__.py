# Assuming we have multiple models in this directory
from .cnn import CNN
from .cnn_sppf import CNN_SPPF
from .cnn_sppf_v2 import CNN_SPPF_v2
# from .model_1 import Model_3

# Making a dictiaonary for convenience
existing_models = {
                    "cnn": CNN,
                    "cnn_sppf": CNN_SPPF,
                    "cnn_sppf_v2": CNN_SPPF_v2
                    # "model_3": Model_3
}

def get_model(model_name, **kwargs):
    if model_name.lower() not in existing_models:
        raise ValueError(f"Model '{model_name}' hasn't been implemented. Choose one of: {list(existing_models.keys())} instead.")
    return existing_models[model_name.lower()](**kwargs)
