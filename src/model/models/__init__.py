from .unet import UNET

# For testing purposes only
_list_of_models = [UNET]


def model_loader(model_name: str):
    return {"unet": UNET}[model_name]
