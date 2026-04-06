
import torch

REPO_DIR = "/Users/trung/Projects/dinov3/"

def load_vits16_model():

    weights  = "/Users/trung/Projects/dino_models/dinov3_vits16_pretrain_lvd1689m.pth"

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


def load_vits16_model_hf():
    """
    Load DINOv3 ViT-L/16 from the Hugging Face Hub.

    Requires:
      - transformers >= 4.56.0
      - Access granted on the HF model page (gated model)
      - HF token set via HUGGINGFACE_HUB_TOKEN env var or huggingface-cli login
    """
    from transformers import AutoImageProcessor, AutoModel
    from huggingface_hub import login
    import os

    MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN", None)

    model = AutoModel.from_pretrained(
        MODEL_ID,
        token=hf_token,       # pass token explicitly if not set globally
    )
    model.eval()
    return model

if __name__ == '__main__':
    model = load_vitl16_model_hf()  # or "cuda" / "auto"
