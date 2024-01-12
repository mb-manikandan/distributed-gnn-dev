import torch


def load_model_dict(path, map_location="cpu"):
    model_state_dict = torch.load(path, map_location=map_location)
    return model_state_dict


def save_model_dict(model_state_dict, path):
    torch.save(model_state_dict, path)

