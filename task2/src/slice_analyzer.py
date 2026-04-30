import numpy as np
from .data_manager import DataManager
from .slice import Slice3D
from .mito_slice_manager import MitoEntry

class SliceAnalyzer:
    """Single-slice reference embedding analysis.

    Owns state for one selected slice and mitochondrion. Use for
    inspecting embedding distances and validating masking behaviour
    before running analysis at dataset scale.
    """

    def __init__(self, data_manager: DataManager):
        """
        Initialize the per slice analyzer with a data manager
        """
        self.data_manager = data_manager
        self.mito_mask = None
        self.reference_vector = None
        self.distance_map = None

    def set_slice(self, slc: Slice3D):
        """Set the slice to analyze."""

        self.slc = slc
        self.seg_data = self.data_manager.segmentation_data.get_slice(slc)
        self.em_data = self.data_manager.em_data.get_slice(slc)

    def set_embeddings(self, embeddings, is_dense: bool):
        """Set the per pixel embeddings for the slice, can be dense
        or patch-level"""

        if is_dense:
            self.embeddings = embeddings
        else:
            from src.embeddings import _upsample_patch_to_dense
            h, w = self.slc.size()
            self.embeddings = _upsample_patch_to_dense(embeddings, h, w)

        self.is_dense = is_dense

    def select_mitochondrion(self, mito_id: int, mito_entry: MitoEntry):
        """Set the active mitochondrion and compute its reference vector.

        Builds a binary mask for mito_id, masks the dense embeddings,
        and averages masked tokens to a single reference vector (D,).
        """
        seg_np = np.array(self.seg_data)
        self.mito_mask = (seg_np == mito_id)
        self.mito_entry = mito_entry

        selected = self.embeddings[0, :] * self.mito_mask  # (D, H, W)
        self.reference_vector = selected.mean((1, 2))       # (D,)

        mito_centroid = (mito_entry.bbox[0]+mito_entry.bbox[1])//2, (mito_entry.bbox[2]+mito_entry.bbox[3])//2
        self.centroid_reference_vector = self.embeddings[0, :, *mito_centroid]

        return self.mito_mask, self.reference_vector

    def compute_distance_map(self, distance_mode='cosine', embedding_mode='centroid') -> np.ndarray:
        """Compute cosine distance from the reference vector to every pixel."""
        from scipy.spatial.distance import cosine
        from numpy.linalg import norm

        if distance_mode == 'cosine':
            dist_func = cosine
        else:
            dist_func = lambda a, b: norm(a - b)  # L2 distance

        H, W = self.embeddings.shape[2], self.embeddings.shape[3]
        flat = self.embeddings[0].reshape((-1, H * W)).T  # (H*W, D)


        if embedding_mode == 'centroid':
            ref_vector = self.centroid_reference_vector
        else:
            ref_vector = self.reference_vector

        self.distance_map = np.array([
            dist_func(flat[i], ref_vector)
            for i in range(H * W)
        ]).reshape(H, W)

        return self.distance_map

    def plot_em_only(self):
        """
        Plot the EM data only for a slice
        """
        import matplotlib.pyplot as plt
        from .visualizer import format_microscopy_ax, compute_extents

        extent = compute_extents(self.data_manager, self.slc)

        fig, ax = plt.subplots(1, 1, figsize=(9, 3))

        ax.imshow(self.em_data, cmap='Grays_r', extent=extent)
        ax.set_title('EM data')
        format_microscopy_ax(ax, self.data_manager, self.slc, grid_alpha=0.5)


    def plot(self, save_path: str = None):
        """Plot EM, selection mask, and distance map side by side.

        Parameters:
            save_path: Optional file path to save the figure before displaying.
        """
        import matplotlib.pyplot as plt
        from .visualizer import format_microscopy_ax, compute_extents

        extent = compute_extents(self.data_manager, self.slc)

        fig, axes = plt.subplots(1, 3, figsize=(9, 3))

        axes[0].imshow(self.em_data, cmap='Grays_r', extent=extent)
        axes[0].set_title('EM data')
        format_microscopy_ax(axes[0], self.data_manager, self.slc)

        axes[1].imshow(self.em_data * self.mito_mask, cmap='Grays_r', extent=extent)
        axes[1].set_title('Selection mask')
        format_microscopy_ax(axes[1], self.data_manager, self.slc)

        axes[2].imshow(self.distance_map, cmap='viridis_r', extent=extent)
        axes[2].set_title('Distances to reference')
        format_microscopy_ax(axes[2], self.data_manager, self.slc)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
