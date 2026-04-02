
import torch
import torch.nn.functional as F

import numpy as np
from .data_manager import DataManager
from .slice import Slice3D
from dinov3.models.vision_transformer import DinoVisionTransformer

class EmbeddingsManager:
    """Computes the embeddings for a dataset"""

    def __init__(self, data_manager: DataManager, model: DinoVisionTransformer):
        self.data_manager = data_manager
        self.model = model

    def compute_dense_embeddings(self, slices):
        """Compute the dense embeddings for slices of the data"""

        from .timer import Timer

        timer = Timer()

        n = len(slices)
        all_dense_maps = []

        for i, slc in enumerate(slices):
            x = _load_data_for_dino(self.data_manager, slc)

            dense_map = _compute_dense_dpt(x, self.model)
            all_dense_maps.append(dense_map)
            
            if i % 20 == 0:
                print(f"{i}/{n} - {timer.get_time()}")


        self.all_dense_maps = all_dense_maps
        return self.all_dense_maps


    def set_slice(self, slc_index, slc):
        """Set slice for reference mitochondria processing"""
        self.slc_index = slc_index # The index of the slice that corresponds to the datasets
        self.slc = slc
        self.slc_segmented_data = self.data_manager.segmentation_data.get_slice(self.slc)
        self.slc_em_data = self.data_manager.em_data.get_slice(self.slc)
        self.slc_embeddings = self.all_dense_maps[self.slc_index]

    def find_mitochondria_ids(self):
        """Retrieve mitochondrial label from the segmentation data"""

        # Select a mitochondria ID in the slice
        segment_ids = list(set(self.slc_segmented_data.flatten()))
        segment_ids = np.array([int(s) for s in segment_ids])
        return segment_ids[segment_ids > 0]

    def select_mitochondria_embeddings(self, sample_segment_id):
        # Create a mask to select the embeddings of the
        # segmented mitochondria alone
        slc_np_data = np.array(self.slc_segmented_data)
        mitochondria_segment_mask = (slc_np_data == sample_segment_id)

        # Keep the state of the segmented mask
        # todo: create another manager for reference embedding calculations
        # that will hold state of selected slices
        self.mitochondria_segment_mask = mitochondria_segment_mask

        # Mask the embeddings with the mitochondria
        # selection
        selected_mitochondria_embeddings = self.slc_embeddings[0, :] * mitochondria_segment_mask

        # Compute an average reference vector
        mitochondria_reference_vector = selected_mitochondria_embeddings.mean((1, 2))

        self.mitochondria_reference_vector = mitochondria_reference_vector
        return selected_mitochondria_embeddings, mitochondria_reference_vector

    def compute_distances_in_slice_to_reference(self):
        """Compute the cosine distance from the reference mitochondria vector
        to every embedding value in the slice."""

        from scipy.spatial.distance import cosine

        W, H = self.slc_embeddings.shape[2], self.slc_embeddings.shape[3]

        flat = self.slc_embeddings[0].reshape((-1, W * H)).T  # (W*H, N)
        ref = self.mitochondria_reference_vector

        result = np.array([
            cosine(flat[i], ref)
            for i in range(W * H)
        ]).reshape(W, H)

        self.slc_embedding_distances = result
        return result

    def plot_distances_to_reference(self):
        import matplotlib.pyplot as plt

        # Plot the mask against the em data to ensure masking is performed properly
        em_slice_data = self.slc_em_data

        fig = plt.figure(figsize=(12, 3))
        plt.subplot(1, 3, 1)
        plt.imshow(em_slice_data, cmap='Grays')
        plt.title("EM data")

        plt.subplot(1, 3, 2)
        plt.imshow(em_slice_data * self.mitochondria_segment_mask, cmap='Grays')
        plt.title("Selection mask")

        plt.subplot(1, 3, 3)
        plt.imshow(self.slc_embedding_distances, cmap='Grays')
        plt.title("Distances to reference")


def _compute_patch_embeddings(model, x):
    """Compute the patch embeddings for an input"""
    import torch
    from PIL import Image
    from torchvision import transforms
    
    with torch.no_grad():
        out = model.forward_features(x)

    return out


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

def _compute_dense_bilinear(x, model):
    """Compute the dense embeddings
    for an output using bilinear interpolation"""

    out_hw = x.shape
    out = _compute_patch_embeddings(self.model, x)

    patch_size = model.patch_size
    w_patches  = W // patch_size
    h_patches  = H // patch_size

    num_embed = model.embed_dim

    patch_tokens = out["x_norm_patchtokens"]
    patch_map = patch_tokens.reshape(1, num_embed, w_patches, h_patches)

    # todo: fix WxH resolution from hard-coded 128
    dense = torch.nn.functional.interpolate(
        patch_map, size=(W, H),
        mode="bilinear", align_corners=False
    )                                                  # (1, D, 128, 128)
    dense = torch.nn.functional.normalize(dense, dim=1)  # L2 norm per pixel
    return dense


def _compute_dense_dpt(img, model):
    """
    Compute dense embeddings using multi-layer DPT-style feature aggregation,
    leveraging DINOv2's built-in get_intermediate_layers().
    """
    W, H = img.shape[2], img.shape[3]

    patch_size = model.patch_size
    w_patches  = W // patch_size
    h_patches  = H // patch_size

    # Collect model blocks, 4 total intermediate layers
    total_blocks = len(model.blocks)
    n_taps = 2
    # indices = [total_blocks * i // n_taps - 1 for i in range(1, n_taps + 1)]

    # Retrieve the last layers
    indices = np.arange(total_blocks-n_taps-1, total_blocks-1, 1)

    # Extract intermediate layers
    intermediates = model.get_intermediate_layers(
        img,
        n=indices,               # Early and latter layers of the model for broad and specific
        return_class_token=True, # (patch_tokens, cls_token) tuples
        reshape=True,            # directly returns (B, D, H_p, W_p) — no reshape needed!
    )

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
