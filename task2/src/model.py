
from transformers import AutoModel
import os


def load_vits16_model_hf():
    """
    Load DINOv3 ViT-L/16 from the Hugging Face Hub.

    Requires:
      - transformers >= 4.56.0
      - Access granted on the HF model page (gated model)
      - HF token set via HUGGINGFACE_HUB_TOKEN env var or huggingface-cli login
    """


    MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN", None)

    model = AutoModel.from_pretrained(
        MODEL_ID,
        token=hf_token,       # pass token explicitly if not set globally
    )
    model.eval()
    return model

if __name__ == '__main__':
    model = load_vits16_model_hf()
    print(type(model))
    print([name for name, _ in model.named_children()])
    print([name for name, _ in model.model.named_children()])
    total_blocks = len(model.model.layer)
    print(total_blocks)
