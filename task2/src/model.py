
import torch

REPO_DIR = "/Users/trung/Projects/dinov3/"

def load_vits16_model():

    weights  = "/Users/trung/Projects/dino_models/dinov3_vitl16_pretrain_lvd1689m.pth"

    model = torch.hub.load(REPO_DIR, "dinov3_vits16", source="local", pretrained=False)
    state_dict = torch.load(weights, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    return model

def load_vitb16_model():

    weights  = "/Users/trung/Projects/dino_models/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"

    model = torch.hub.load(REPO_DIR, "dinov3_vitb16", source="local", pretrained=False)
    state_dict = torch.load(weights, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    return model

def load_vith16plus_model():

    weights  = "/Users/trung/Projects/dino_models/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"

    model = torch.hub.load(REPO_DIR, "dinov3_vith16plus", source="local", pretrained=False)
    state_dict = torch.load(weights, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    return model
