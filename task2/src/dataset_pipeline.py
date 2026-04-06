from typing import Dict, List, Optional
import numpy as np
from transformers import PreTrainedModel

from .data_manager import DataManager
from .slice import Slice3D
from .mito_slice_manager import MitoEntry
from .visualizer import Visualizer
from .reference_analyzer import ReferenceAnalyzer

class DatasetPipeline:
    """End-to-end single-dataset mitochondria analysis pipeline.

    Orchestrates the full workflow from data loading through per-mito embedding
    vectors.

    Call run() to execute all steps in sequence.

    Attributes:
        name: Human-readable label for this dataset (used in plots/logs).
        data_manager: Loaded DataManager for this dataset.
        original_slices: Full set of candidate patches from SliceGenerator (all z-planes).
        slices: Working subset of patches used for catalog building and downstream steps.
            Set to a random sample of original_slices in build_catalog() for faster
            iteration; all subsequent steps (embeddings, mito vectors) operate on this
            fixed set so slice indices remain consistent across steps.
        mito_catalog: Dict mapping mito_id → MitoEntry (best slice per mito), built
            from self.slices.
        all_patch_embeddings: List of patch embedding arrays (1, D, H_p, W_p),
            one per entry in self.slices, in the same order.
        all_mito_vectors: Dict mapping mito_id → reference embedding vector (D,).
    """

    def __init__(
        self,
        group_url: str,
        seg_url: str,
        em_resolution: str,
        segmentation_resolution: str,
        model: PreTrainedModel,
        name: str = "",
        patch_size: int = 512,
        z_step: int = 1,
        inset: int = 0,
        min_pixels: int = 100,
        boundary_margin: int = 8,
        output_dir: str = './output',
        num_random_samples: int = 10
    ):
        """
        Parameters:
            group_url: S3 path to the EM data group (N5/Zarr).
            seg_url: S3 path to the segmentation data group.

            em_resolution: Resolution key for EM data (e.g. 's1').
            segmentation_resolution: Resolution key for segmentation (e.g. 's0').

            model: Pre-loaded DINOv2 model shared across pipelines.
            name: Dataset label used in logs and plots.
            
            patch_size: Spatial size (px) of each generated patch.
            z_step: Step size along z when generating patches.
            inset: Pixel inset from volume boundaries when generating patches.
            min_pixels: Minimum mito area (px) to retain in the catalog.
            boundary_margin: Pixels from patch edge within which mitos are excluded.
        """
        self.name = name
        self.model = model
        self.patch_size = patch_size
        self.z_step = z_step
        self.inset = inset
        self.min_pixels = min_pixels
        self.boundary_margin = boundary_margin
        self.output_dir = output_dir
        self.num_random_samples = num_random_samples

        self.data_manager: Optional[DataManager] = None
        self.original_slices: Optional[List[Slice3D]] = None
        self.slices: Optional[List[Slice3D]] = None
        self.mito_catalog: Optional[Dict[int, MitoEntry]] = None
        self.all_patch_embeddings: Optional[List[np.ndarray]] = None
        self.all_mito_vectors: Optional[Dict[int, np.ndarray]] = None

        self._group_url = group_url
        self._seg_url = seg_url
        self._em_resolution = em_resolution
        self._segmentation_resolution = segmentation_resolution

    # ------------------------------------------------------------------
    # Individual pipeline steps
    # ------------------------------------------------------------------

    def load_data(self) -> DataManager:
        """Step 1 — Instantiate and return the DataManager for this dataset.

        Connects to the S3-hosted EM and segmentation arrays at the configured
        resolutions. Result stored in self.data_manager.
        """
        self.data_manager = DataManager(
            self._group_url,
            self._seg_url,
            self._em_resolution,
            self._segmentation_resolution,
            self.name,
        )
        return self.data_manager

    def generate_slices(self) -> List[Slice3D]:
        """Step 2 — Generate all candidate patches across the volume.

        Uses SliceGenerator with the configured patch_size, z_step, and inset.
        Result stored in self.original_slices. self.slices is also set to the
        full list here and may be narrowed to a random sample in build_catalog().
        """
        from .slice_generator import SliceGenerator

        generator = SliceGenerator(
            self.data_manager,
            patch_size=self.patch_size,
            z_step=self.z_step,
            inset=self.inset,
        )
        self.original_slices = generator.generate()
        self.slices = self.original_slices
        return self.slices

    def build_catalog(self) -> Dict[int, MitoEntry]:
        """Step 3 — Scan a random sample of patches and build the mito catalog.

        Randomly samples num_slices_search patches from original_slices (seeded for
        reproducibility) and stores them as self.slices. This working set is then
        used for catalog building, embedding computation, and mito vector extraction,
        keeping slice indices consistent across all downstream steps.

        Uses MitoSliceManager with the configured min_pixels and boundary_margin.
        Result stored in self.mito_catalog.
        """
        from .mito_slice_manager import MitoSliceManager

        print(f"Randomly selecting {self.num_random_samples} slices for quick analysis")
        num_slices_search = self.num_random_samples
        np.random.seed(123)
        slice_indices = np.random.choice(len(self.original_slices), size=num_slices_search)
        self.slices = [self.original_slices[i] for i in slice_indices]

        manager = MitoSliceManager(
            self.data_manager,
            self.slices,
            min_pixels=self.min_pixels,
            boundary_margin=self.boundary_margin,
        )
        self.mito_catalog = manager.build()

        return self.mito_catalog

    def compute_embeddings(self) -> List[np.ndarray]:
        """Step 4 — Compute patch-resolution embeddings for slices.

        Uses EmbeddingsManager.compute_patch_embeddings() over self.slices.
        Result stored in self.all_patch_embeddings.
        """
        from .embeddings import EmbeddingsManager

        embeddings_manager = EmbeddingsManager(self.data_manager, self.model)
        self.all_patch_embeddings = embeddings_manager.compute_patch_embeddings(self.slices)
        return self.all_patch_embeddings

    def build_mito_vectors(self) -> Dict[int, np.ndarray]:
        """Step 5 — Compute per-mito reference embedding vectors.

        Uses MitoEmbeddingBuilder over self.mito_catalog, self.all_patch_embeddings,
        and self.slices. Result stored in self.all_mito_vectors.
        """
        from .mito_embedding_builder import MitoEmbeddingBuilder
        from .slice_analyzer import SliceAnalyzer

        slice_analyzer = SliceAnalyzer(self.data_manager)
        builder = MitoEmbeddingBuilder(
            mito_catalog=self.mito_catalog,
            all_patch_embeddings=self.all_patch_embeddings,
            slices=self.slices,
            slice_analyzer=slice_analyzer,
        )
        self.all_mito_vectors = builder.build()
        return self.all_mito_vectors

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def run(self):
        """Execute all pipeline steps in sequence (steps 1–5)."""
        print("Loading EM and segmentation data...")
        self.load_data()

        print("Generating slices...")
        self.generate_slices()

        print("Building mitochondria to slices catalog...")
        self.build_catalog()

        print("Computing embeddings for each slice....")
        self.compute_embeddings()

        print("Computing mitochondrial embeddings....")
        self.build_mito_vectors()

        print("Running single dataset analysis....")
        self.vis = Visualizer(self.data_manager)
        self.reference_analyzer = ReferenceAnalyzer(self, self.vis)
        self.reference_analyzer.run(reference_mito_id_index=2)

    def mito_ids(self) -> List[int]:
        """Return mito IDs retained in the catalog."""
        return list(self.mito_catalog.keys())

    def __repr__(self):
        n_mitos = len(self.mito_catalog) if self.mito_catalog is not None else "?"
        n_slices = len(self.slices) if self.slices is not None else "?"
        n_original = len(self.original_slices) if self.original_slices is not None else "?"
        return f"DatasetPipeline(name={self.name!r}, slices={n_slices}/{n_original}, mitos={n_mitos})"
