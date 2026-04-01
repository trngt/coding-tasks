
import torch
import numpy as np
from .data_manager import DataManager
from dinov3.models.vision_transformer import DinoVisionTransformer

class EmbeddingsManager:
    """Computes the embeddings for a dataset"""

    def __init__(self, data_manager: DataManager, model: DinoVisionTransformer):
        self.data_manager = data_manager
        self.model = model

    def compute_dense_embeddings(self, slices):

        n = len(slices)

        embeddings = []
        all_dense_maps = []

        for i, slc in enumerate(slices):
            x = _load_data_for_dino(self.data_manager, slc)
            out = _compute_patch_embeddings(self.model, x)
            embeddings.append(out)

            dense_map = _compute_dense(out)
            all_dense_maps.append(dense_map)
            
            if i % 2 == 0:
                print(f"{i}/{n}")

            if i == 10:
                # todo: debugging against first 10 slices
                break

        self.all_dense_maps = all_dense_maps
        return self.all_dense_maps


    def set_slice(self, slc_index, slc):
        """Set slice for reference mitochondria processing"""
        self.slc_index = slc_index # The index of the slice that corresponds to the datasets
        self.slc = slc
        self.slc_segmented_data = self.data_manager.segmentation_data.get_slice(self.slc)
        self.slc_em_data = self.data_manager.em_data.get_slice(self.slc)
        self.slc_embeddings = self.all_dense_maps[self.slc_index].numpy()

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
    arr = img.values  # (128, 128) numpy array
    
    # Normalize to [0, 1] first if your data isn't already
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    
    # Replicate to 3 channels and convert to tensor: (1, 3, 128, 128)
    arr_3ch = np.stack([arr, arr, arr], axis=0)          # (3, 128, 128)
    x = torch.tensor(arr_3ch, dtype=torch.float32).unsqueeze(0)  # (1, 3, 128, 128)
    
    # Apply ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x

def _compute_dense(out):
    """Compute the dense embeddings
    for an output.

    todo: Fixed at 384x8x8 per the VIT-16S model."""

    patch_tokens = out["x_norm_patchtokens"]          # (1, 64, 384)
    patch_map = patch_tokens.reshape(1, 384, 8, 8)    # (1, 384, 8, 8)

    dense = torch.nn.functional.interpolate(
        patch_map, size=(128, 128),
        mode="bilinear", align_corners=False
    )                                                  # (1, 384, 128, 128)
    dense = torch.nn.functional.normalize(dense, dim=1)  # L2 norm per pixel
    return dense