import numpy as np
from .slice_analyzer import SliceAnalyzer


class ReferenceAnalyzer:
    """Compares all catalogued mitochondria against a chosen reference in embedding space.

    Owns the reference state (mito id, vector, distance DataFrame) and provides
    methods for distance computation, distribution plotting, and visual inspection
    of closest/furthest examples.

    Attributes:
        reference_mito_id: The currently selected reference mito id.
        reference_vector: Embedding vector (D,) of the reference mito.
        distances_df: DataFrame indexed by mito_id with column 'l2_distance',
            sorted ascending after compute_distances() is called.
    """

    def __init__(self, pipeline, vis):
        """
        Parameters:
            pipeline: A fully run DatasetPipeline instance. Must have mito_catalog,
                all_mito_vectors, all_patch_embeddings, and slices populated.
            vis: Visualizer instance with plot_mito_mask().
        """
        self.pipeline = pipeline
        self.vis = vis
        self.slice_analyzer = SliceAnalyzer(pipeline.data_manager)

        self.reference_mito_id = None
        self.reference_vector: np.ndarray = None
        self.distances_df = None

    def set_reference(self, mito_id: int):
        """Select a reference mitochondrion and compute its per-pixel distance map.

        Loads the slice embeddings into the shared SliceAnalyzer, selects the mito
        to set the reference vector, and computes the distance map for visualization.

        Parameters:
            mito_id: ID of the mitochondrion to use as the reference.
        """
        self.reference_mito_id = mito_id
        self.reference_vector = self.pipeline.all_mito_vectors[mito_id]

        entry = self.pipeline.mito_catalog[mito_id]
        slc = self.pipeline.slices[entry.slice_index]
        patch_embeddings = self.pipeline.all_patch_embeddings[entry.slice_index]

        self.slice_analyzer.set_slice(slc)
        self.slice_analyzer.set_embeddings(patch_embeddings, is_dense=False)
        self.slice_analyzer.select_mitochondrion(mito_id)
        self.slice_analyzer.compute_distance_map()

    def plot_reference(self):
        """Plot EM image, selection mask, and distance map for the reference mito."""
        self.slice_analyzer.plot()

    def compute_distances(self):
        """Compute L2 distances from the reference vector to all mito vectors.

        Populates self.distances_df sorted by ascending distance.

        Returns:
            DataFrame indexed by mito_id with column 'l2_distance'.
        """
        import pandas as pd
        from numpy.linalg import norm

        distances = {
            mito_id: norm(self.reference_vector - vec)
            for mito_id, vec in self.pipeline.all_mito_vectors.items()
        }
        self.distances_df = (
            pd.DataFrame(distances.items(), columns=["mito_id", "l2_distance"])
            .set_index("mito_id")
            .sort_values("l2_distance")
        )
        return self.distances_df

    def plot_distance_distribution(self):
        """Plot a histogram of L2 distances to the reference mito."""
        import matplotlib.pyplot as plt

        n = len(self.distances_df)
        plt.figure(figsize=(4, 2.5))
        plt.hist(self.distances_df.l2_distance, bins=50)
        plt.title(
            f"Mitochondria embedding\ndistances to reference, n={n}",
            fontsize=16,
            fontweight="demi",
        )
        plt.xlabel("L2 distance")
        plt.tight_layout()
        plt.show()

    def closest(self, n: int = 10):
        """Return the n closest mitos (excluding the reference itself).

        Parameters:
            n: Number of entries to return.

        Returns:
            Slice of distances_df with the n smallest distances.
        """
        return self.distances_df.iloc[1 : n + 1]

    def furthest(self, n: int = 10):
        """Return the n furthest mitos.

        Parameters:
            n: Number of entries to return.

        Returns:
            Slice of distances_df with the n largest distances.
        """
        return self.distances_df.dropna().tail(n)

    def plot_closest(self, k=10):
        self.plot_series(self.closest(k), title=f"{k} nearest mitochondria")

    def plot_furthest(self, k=10):
        self.plot_series(self.furthest(k), title=f"{k} furthest mitochondria")

    def plot_series(self, entries, pad_size: int = 200, title=""):
        """Plot a row of representative mito images for the given distance entries.

        For each mito, pads its tight bounding-box slice to pad_size × pad_size and
        renders it with the mito mask highlighted.

        Parameters:
            entries: DataFrame slice (from closest() or furthest()) iterated by mito_id.
            pad_size: Target height and width in pixels for each thumbnail.
        """
        import matplotlib.pyplot as plt
        from .math_helpers import pad_slice_to_size

        n = len(entries)
        fig, axs = plt.subplots(1, n, figsize=(11, 2))
        axs = np.array(axs).flatten()

        for i, (plot_mito_id, _) in enumerate(entries.iterrows()):
            ax = axs[i]
            entry = self.pipeline.mito_catalog[plot_mito_id]
            slc = entry.to_slice()

            try:
                slc = pad_slice_to_size(slc, pad_size, pad_size)
            except ValueError:
                print(f"Slice {slc}, exceeds padding size parameter, reverting to plot entire slice.")
                pass

            z_index = slc.z.start
            self.vis.plot_mito_mask(slc, z_index=z_index, highlight_mito_id=plot_mito_id, ax=ax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel("")
            ax.set_xlabel("")
            ax.set_title(f"{i}: {plot_mito_id}", fontsize=7)

        plt.suptitle(title, fontsize=16, fontweight='demi')
        plt.tight_layout()
        plt.show()

    def run(self, reference_mito_id_index=0):
        """Run the entire workflow for a single dataset reference
        mitochondrion workflow"""

        # Select a reference mitochondrion
        self.set_reference(self.pipeline.mito_ids()[reference_mito_id_index])
        self.compute_distances()
        self.plot_distance_distribution()
        self.plot_reference()
        self.plot_closest()
        self.plot_furthest()
