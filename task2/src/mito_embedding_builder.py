from typing import Dict, List
import numpy as np
from .mito_slice_manager import MitoEntry
from .slice_analyzer import SliceAnalyzer


class MitoEmbeddingBuilder:
    """Computes and stores a reference embedding vector for every catalogued mitochondrion.

    Iterates the mito catalog in order, reusing the current slice's upsampled embeddings
    when consecutive mitos share the same slice (avoiding redundant upsampling).

    Attributes:
        all_mito_vectors: Dict mapping mito_id → reference vector (D,) after build().
    """

    def __init__(
        self,
        mito_catalog: Dict[int, MitoEntry],
        all_patch_embeddings: List[np.ndarray],
        slices: list,
        slice_analyzer: SliceAnalyzer,
    ):
        """
        Parameters:
            mito_catalog: Mapping of mito_id → MitoEntry (from MitoSliceManager.catalog).
            all_patch_embeddings: List of patch embedding arrays (1, D, H_p, W_p),
                indexed by slice_index (as returned by EmbeddingsManager.compute_patch_embeddings).
            slices: The slice list used to produce all_patch_embeddings
                (e.g. slices_subset from SliceGenerator).
            slice_analyzer: Shared SliceAnalyzer instance used to upsample and mask embeddings.
        """
        self.mito_catalog = mito_catalog
        self.all_patch_embeddings = all_patch_embeddings
        self.slices = slices
        self.slice_analyzer = slice_analyzer
        self.all_mito_vectors: Dict[int, np.ndarray] = {}

    def build(self) -> Dict[int, np.ndarray]:
        """Compute reference vectors for all mitos in the catalog.

        For each mito, sets the slice and embeddings on the shared SliceAnalyzer only
        when the slice changes, then calls select_mitochondrion to obtain the masked
        mean embedding vector. Progress is printed every 10 mitos.

        Returns:
            self.all_mito_vectors: mito_id → reference vector (D,).
        """
        from src.timer import Timer
        from tqdm import tqdm

        timer = Timer()
        self.all_mito_vectors = {}
        last_slice_index = None
        mito_ids = list(self.mito_catalog.keys())

        with tqdm(mito_ids, desc="Computing mito vectors") as pbar:
            for mito_id in pbar:
                entry = self.mito_catalog[mito_id]

                if entry.slice_index != last_slice_index:
                    slc = self.slices[entry.slice_index]
                    patch_embeddings = self.all_patch_embeddings[entry.slice_index]
                    self.slice_analyzer.set_slice(slc)
                    self.slice_analyzer.set_embeddings(patch_embeddings, is_dense=False)
                    last_slice_index = entry.slice_index

                self.slice_analyzer.select_mitochondrion(mito_id)
                self.all_mito_vectors[mito_id] = self.slice_analyzer.reference_vector
                pbar.set_postfix(elapsed=timer.get_time())

        return self.all_mito_vectors
