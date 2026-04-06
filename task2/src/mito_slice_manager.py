from typing import Dict, List
import numpy as np
from .data_manager import DataManager
from .slice import Slice3D


class MitoEntry:
    """Holds the best-slice record for a single mitochondrion.

    For this analysis, we'll assume a single mitochondria will be entirely
    contained in a single slice of the data. 

    todo: the reality is, we have an entire
    volume of mitochondria data, so we'd want to change the structure of
    this workflow for a complete solution.

    """

    def __init__(self, mito_id: int, slice_index: int, slc: Slice3D,
                 num_pixels: int, bbox: tuple):
        self.mito_id = mito_id
        self.slice_index = slice_index  # index into the original slices list
        self.slc = slc
        self.z = slc.z.start            # absolute z index in the volume
        self.num_pixels = num_pixels    # pixels occupied by this mito in slc
        self.bbox = bbox                # (y_min, y_max, x_min, x_max) in patch-local coords

    def to_slice(self) -> Slice3D:
        """Create a tight Slice3D around this mitochondrion.

        Converts the patch-local bbox coordinates to absolute volume
        coordinates by offsetting with the parent patch origin.
        """
        y_min, y_max, x_min, x_max = self.bbox
        abs_y_min = self.slc.y.start + y_min
        abs_y_max = self.slc.y.start + y_max + 1  # +1 for exclusive slice end
        abs_x_min = self.slc.x.start + x_min
        abs_x_max = self.slc.x.start + x_max + 1
        return Slice3D(
            z=slice(self.z, self.z),
            y=slice(abs_y_min, abs_y_max),
            x=slice(abs_x_min, abs_x_max),
        )

    def __repr__(self):
        return (f"MitoEntry(id={self.mito_id}, z={self.z}, slice_index={self.slice_index}, "
                f"num_pixels={self.num_pixels}, bbox={self.bbox})")


class MitoSliceManager:
    """Builds a catalog mapping each mitochondrion to its best representative slice.

    Iterates over all patches produced by SliceGenerator, scans the segmentation
    data for each patch, and keeps — per mito ID — the slice with the greatest
    num pixels. Mitos below min_pixels are discarded.

    Attributes:
        catalog: Dict mapping mito_id → MitoEntry for the best slice found.
    """

    def __init__(self, data_manager: DataManager, slices: List[Slice3D],
                 min_pixels: int = 1024, boundary_margin: int = 8):
        """
        Parameters:
            data_manager: Provides access to segmentation data.
            slices: All candidate patches from SliceGenerator.generate().
            min_pixels: Minimum number of pixels a mito must occupy in its
                best slice to be retained. 1024 ~ 4 total patches (16x16)
            boundary_margin: Pixels from the patch edge within which a mito
                is considered a boundary mito and excluded. Default is 8
                (half a ViT patch size of 16).
        """
        self.data_manager = data_manager
        self.slices = slices
        self.min_pixels = min_pixels
        self.boundary_margin = boundary_margin
        self.catalog: Dict[int, MitoEntry] = {}

    def build(self) -> Dict[int, MitoEntry]:
        from src.timer import Timer
        from tqdm import tqdm

        timer = Timer()
        self.catalog = {}

        with tqdm(enumerate(self.slices), desc="", total=len(self.slices)) as pbar:
            for slice_index, slc in pbar:

                seg_patch = self.data_manager.segmentation_data.get_slice(slc)
                entries = self._compute_mito_stats(seg_patch, slice_index, slc)
                H, W = seg_patch.shape

                for entry in entries:
                    y_min, y_max, x_min, x_max = entry.bbox
                    m = self.boundary_margin
                    is_boundary = (
                        y_min <= m or y_max >= H - 1 - m or
                        x_min <= m or x_max >= W - 1 - m
                    )
                    if is_boundary:
                        continue

                    existing = self.catalog.get(entry.mito_id)
                    if (entry.num_pixels >= self.min_pixels) and \
                       (existing is None or entry.num_pixels > existing.num_pixels):
                        self.catalog[entry.mito_id] = entry

                pbar.set_postfix(elapsed=timer.get_time(), catalog_size=len(self.catalog))

        return self.catalog

    def _compute_mito_stats(self, seg_patch: np.ndarray, slice_index: int,
                            slc: Slice3D) -> List[MitoEntry]:
        """Return a MitoEntry for each non-background mito in seg_patch.

        Parameters:
            seg_patch: 2D segmentation array (H, W) with integer mito IDs.
            slice_index: Index of this patch in self.slices.
            slc: The Slice3D corresponding to seg_patch.
        """
        total_pixels = seg_patch.size
        mito_ids, pixel_counts = np.unique(seg_patch, return_counts=True)

        entries = []
        for mito_id, num_pixels in zip(mito_ids, pixel_counts):

            # Skip background
            if mito_id == 0:
                continue

            rows, cols = np.where(seg_patch == mito_id)
            bbox = (int(rows.min()), int(rows.max()),
                    int(cols.min()), int(cols.max()))  # (y_min, y_max, x_min, x_max)

            entries.append(MitoEntry(
                mito_id=int(mito_id),
                slice_index=slice_index,
                slc=slc,
                num_pixels=int(num_pixels),
                bbox=bbox,
            ))

        return entries

    def best_slices(self) -> List[Slice3D]:
        """Return the unique set of slices covering at least one catalogued mito."""
        seen = set()
        slices = []
        for entry in self.catalog.values():
            if entry.slice_index not in seen:
                seen.add(entry.slice_index)
                slices.append(entry.slc)
        return slices

    def mito_ids(self) -> List[int]:
        """Return all mito IDs retained in the catalog."""
        return list(self.catalog.keys())

    def plot_size_distribution(self):
        """Plot histogram of mito areas in µm² across the catalog."""
        import matplotlib.pyplot as plt

        dimensions = self.data_manager.segmentation_data.data.attrs['pixelResolution']['dimensions']
        x_nm_per_pixel, y_nm_per_pixel, _ = tuple(dimensions)
        um2_per_pixel = (x_nm_per_pixel / 1000) * (y_nm_per_pixel / 1000)

        areas_um2 = [entry.num_pixels * um2_per_pixel for entry in self.catalog.values()]
        min_pixels_um2 = self.min_pixels * um2_per_pixel

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(areas_um2, bins=50, edgecolor='black')
        ax.set_xlabel('Area (µm²) (pixels)')
        ax.set_ylabel('Count')
        ax.set_title(f'Mitochondria 2D size distribution (n={len(areas_um2)})')

        # Secondary x-axis ticks showing pixel counts
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim()[0] / um2_per_pixel, ax.get_xlim()[1] / um2_per_pixel)
        ax2.set_xlabel('Pixels')

        ax.axvline(min_pixels_um2, color='red', linestyle='--', label=f'min={min_pixels_um2:.3f} µm² ({self.min_pixels} px)')
        ax.legend()
        plt.tight_layout()
        plt.show()
