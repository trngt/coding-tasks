
import torch
import torch.nn.functional as F

import numpy as np
from typing import TYPE_CHECKING, List
from .data_manager import DataManager
from .slice import Slice3D

from transformers import PreTrainedModel


class EmbeddingsManager:
    """Computes embeddings for a dataset."""

    def __init__(self, data_manager: DataManager, model: PreTrainedModel):
        self.data_manager = data_manager
        self.model = model

    def compute_dense_embedding(self, slc: Slice3D) -> np.ndarray:
        """Workflow 2: Compute full-resolution dense embedding for a single slice.

        Returns (1, D, H, W). Not stored — caller owns the result.
        Use for visualization and per-slice inspection.
        """
        x = _load_data_for_dino(self.data_manager, slc)
        return _compute_dense_embeddings(x, self.model)

    def compute_patch_embedding(self, slc: Slice3D) -> np.ndarray:
        """Compute patch-resolution embedding for a single slice.

        Returns (1, D, H_p, W_p). Not stored — caller owns the result.
        """
        x = _load_data_for_dino(self.data_manager, slc)
        return _compute_patch_embeddings(x, self.model)

    def compute_patch_embeddings(self, slices: List[Slice3D]) -> List[np.ndarray]:
        """Workflow 1: Compute patch-resolution embeddings for all slices.

        Returns a list of (1, D, H_p, W_p) arrays held in memory.
        """
        from .timer import Timer
        from tqdm import tqdm

        timer = Timer()
        n = len(slices)
        all_patch_maps = []

        for slc in (pbar := tqdm(slices, desc="Computing patch embeddings")):
            x = _load_data_for_dino(self.data_manager, slc)
            patch_map = _compute_patch_embeddings(x, self.model)
            all_patch_maps.append(patch_map)
            pbar.set_postfix(elapsed=timer.get_time())

        self.all_patch_maps = all_patch_maps
        return self.all_patch_maps


def _upsample_patch_to_dense(patch_embedding: np.ndarray,
                            target_h: int, target_w: int) -> np.ndarray:
    """Upsample a patch-resolution embedding map to dense (pixel) resolution.

    Parameters:
        patch_embedding: Array of shape (1, D, H_p, W_p).
        target_h: Target height in pixels (H of the original image).
        target_w: Target width in pixels (W of the original image).

    Returns:
        Dense embedding array of shape (1, D, target_h, target_w).
    """
    t = torch.from_numpy(patch_embedding)
    upsampled = F.interpolate(t, size=(target_h, target_w),
                              mode='bilinear', align_corners=False)
    upsampled = F.normalize(upsampled, dim=1)
    return upsampled.numpy()


def _load_data_for_dino(data_manager, slc):
    # Run DINOv3 on the image slices to compute embeddings

    img = data_manager.em_data.data.isel(
        z=slc.z.start,
        y=slc.y,
        x=slc.x,
    ).compute()

    # da is your xarray DataArray with shape (H, W)
    arr = img.values  # (W, H) numpy array

    # Normalize to [0, 1] first if your data isn't already
    arr = (arr - arr.min()) / (arr.max() - arr.min())

    # Replicate to 3 channels and convert to tensor: (1, 3, W, H)
    arr_3ch = np.stack([arr, arr, arr], axis=0)          # (3, W, H)
    x = torch.tensor(arr_3ch, dtype=torch.float32).unsqueeze(0)  # (1, 3, W, H)

    # Apply ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x


def _compute_patch_embeddings(img, model):
    """
    Compute patch-level embeddings (with option to aggregate intermediate layers)
    """
    total_blocks = len(model.model.layer)
    n_taps = 1
    indices = np.arange(total_blocks - n_taps - 1, total_blocks - 1, 1)

    intermediates = get_model_intermediates(img, model, n_taps)

    layer_maps = [patch_map for patch_map, _cls in intermediates]

    # Average the layers (applicable if multiple layers are retrieved)
    patch_map = torch.stack(layer_maps, dim=0).mean(dim=0)  # (B, D, H_p, W_p)
    patch_map = F.normalize(patch_map, dim=1)

    return patch_map.detach().numpy()


def _compute_dense_embeddings(img, model):
    """
    Compute dense embeddings using a simple bilinear interpolation.
    """
    W, H = img.shape[2], img.shape[3]

    patch_size = model.config.patch_size
    w_patches  = W // patch_size
    h_patches  = H // patch_size

    # Collect model blocks, last layer
    total_blocks = len(model.model.layer)
    n_taps = 1

    # Retrieve the n last layers (1-4 appear to have similar results with this simple
    # interpolation method)
    indices = np.arange(total_blocks-n_taps-1, total_blocks-1, n_taps)

    # Extract intermediate layers
    intermediates = get_model_intermediates(img, model, n_taps)

    # Upsample each layer's map to the target resolution
    layer_maps = []
    for patch_map, _cls in intermediates:
        upsampled = F.interpolate(
            patch_map, size=(W, H),
            mode="bilinear", align_corners=False
        )
        layer_maps.append(upsampled)

    # Fuse by averaging
    dense = torch.stack(layer_maps, dim=0).mean(dim=0)    # (B, D, H, W)

    # normalize per pixel
    dense = F.normalize(dense, dim=1)

    return dense.detach().numpy()


def get_model_intermediates(img, model, n_taps=1):
    """
    Returns intermediate patch maps from the HF DINOv3ViTModel,
    matching the (patch_map, cls_token) tuple format of get_intermediate_layers().

    Handles models with register tokens by taking the last H_p*W_p tokens
    as patch tokens, regardless of any prefix tokens (CLS, registers, etc.)

    Returns:
        List of (patch_map, cls_token) tuples, where:
            patch_map: (B, D, H_p, W_p)
            cls_token: (B, D)
    """
    total_blocks = len(model.model.layer)
    indices = np.arange(total_blocks - n_taps - 1, total_blocks - 1, 1)

    B = img.shape[0]
    H_p = img.shape[2] // model.config.patch_size
    W_p = img.shape[3] // model.config.patch_size
    n_patch_tokens = H_p * W_p

    with torch.no_grad():
        outputs = model(img, output_hidden_states=True)

    intermediates = []
    for i in indices:
        hs = outputs.hidden_states[i + 1]           # (B, 1+registers+H_p*W_p, D)
        cls_token = hs[:, 0, :]                     # (B, D)
        patch_tokens = hs[:, -n_patch_tokens:, :]   # (B, H_p*W_p, D)
        D = patch_tokens.shape[-1]
        patch_map = patch_tokens.permute(0, 2, 1).reshape(B, D, H_p, W_p)
        intermediates.append((patch_map, cls_token))

    return intermediates
