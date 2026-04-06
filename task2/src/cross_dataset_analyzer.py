import numpy as np
import matplotlib.pyplot as plt
from .slice_analyzer import SliceAnalyzer
from .visualizer import Visualizer


class CrossDatasetAnalyzer:
    """Compares mitochondria in a target dataset against a reference from a source dataset.

    The reference vector is drawn from source_pipeline; distances are computed
    against all mito vectors in both source_pipeline and target_pipeline.
    Visualization of reference context (EM + mask + distance map) uses the source
    dataset; series thumbnails use the target dataset.

    Attributes:
        color_source: Plot color for source dataset distances.
        color_target: Plot color for target dataset distances.
        reference_mito_id: The currently selected reference mito id (from source).
        reference_vector: Embedding vector (D,) of the reference mito.
        distances_source_df: DataFrame of L2 distances to all mitos in source_pipeline.
        distances_df: DataFrame of L2 distances to all mitos in target_pipeline.
        distances_combined_df: DataFrame of L2 distances to mitos from both pipelines,
            with an 'origin' column indicating the source dataset name.
    """

    def __init__(self, source_pipeline, target_pipeline):
        """
        Parameters:
            source_pipeline: DatasetPipeline from which the reference mito is drawn.
            target_pipeline: DatasetPipeline whose mito population is compared against
                the reference.
        """
        self.source_pipeline = source_pipeline
        self.target_pipeline = target_pipeline

        self.slice_analyzer = SliceAnalyzer(source_pipeline.data_manager)
        self.vis_source = Visualizer(source_pipeline.data_manager)
        self.vis_target = Visualizer(target_pipeline.data_manager)

        self.color_source = plt.cm.Reds(0.5)
        self.color_target = plt.cm.Blues(0.5)

        self.reference_mito_id = None
        self.reference_vector: np.ndarray = None
        self.distances_source_df = None
        self.distances_df = None
        self.distances_combined_df = None

    def set_reference(self, mito_id: int):
        """Select a reference mitochondrion from the source dataset.

        Loads the source slice embeddings, selects the mito to set the reference
        vector, and computes the per-pixel distance map for visualization.

        Parameters:
            mito_id: ID of the mitochondrion in source_pipeline to use as reference.
        """
        self.reference_mito_id = mito_id
        self.reference_vector = self.source_pipeline.all_mito_vectors[mito_id]

        entry = self.source_pipeline.mito_catalog[mito_id]
        slc = self.source_pipeline.slices[entry.slice_index]
        patch_embeddings = self.source_pipeline.all_patch_embeddings[entry.slice_index]

        self.slice_analyzer.set_slice(slc)
        self.slice_analyzer.set_embeddings(patch_embeddings, is_dense=False)
        self.slice_analyzer.select_mitochondrion(mito_id)
        self.slice_analyzer.compute_distance_map()

    def plot_reference(self):
        """Plot EM image, selection mask, and distance map for the reference mito
        in the context of the source dataset."""
        self.slice_analyzer.plot()

    def compute_distances(self):
        """Compute L2 distances from the reference vector to all mitos in both pipelines.

        Populates self.distances_source_df (source population) and self.distances_df
        (target population), both sorted by ascending distance.

        Returns:
            (distances_source_df, distances_df) tuple.
        """
        import pandas as pd
        from numpy.linalg import norm

        def _compute(mito_vectors):
            distances = {
                mito_id: norm(self.reference_vector - vec)
                for mito_id, vec in mito_vectors.items()
            }
            return (
                pd.DataFrame(distances.items(), columns=["mito_id", "l2_distance"])
                .set_index("mito_id")
                .sort_values("l2_distance")
            )

        self.distances_source_df = _compute(self.source_pipeline.all_mito_vectors)
        self.distances_df = _compute(self.target_pipeline.all_mito_vectors)
        return self.distances_source_df, self.distances_df

    def compute_combined_distances(self):
        """Compute L2 distances against the merged population of both pipelines.

        Entries are tagged with their origin dataset name in an 'origin' column.
        Populates self.distances_combined_df sorted by ascending distance.

        Note: mito_ids are assumed to be unique across both datasets. If the same
        integer ID exists in both, the source entry takes precedence.

        Returns:
            distances_combined_df with columns ['l2_distance', 'origin'].
        """
        import pandas as pd
        from numpy.linalg import norm

        rows = []
        for mito_id, vec in self.source_pipeline.all_mito_vectors.items():
            rows.append((mito_id, norm(self.reference_vector - vec), self.source_pipeline.name))
        for mito_id, vec in self.target_pipeline.all_mito_vectors.items():
            rows.append((mito_id, norm(self.reference_vector - vec), self.target_pipeline.name))

        self.distances_combined_df = (
            pd.DataFrame(rows, columns=["mito_id", "l2_distance", "origin"])
            .set_index("mito_id")
            .sort_values("l2_distance")
        )
        return self.distances_combined_df

    def plot_distance_distribution(self):
        """Plot overlaid histograms of L2 distances for both source and target populations."""
        n_source = len(self.distances_source_df)
        n_target = len(self.distances_df)

        plt.figure(figsize=(5, 3))
        plt.hist(self.distances_source_df.l2_distance, bins=50, alpha=0.6,
                 color=self.color_source, label=f"{self.source_pipeline.name} (n={n_source})")
        plt.hist(self.distances_df.l2_distance, bins=50, alpha=0.6,
                 color=self.color_target, label=f"{self.target_pipeline.name} (n={n_target})")
        plt.title("Distances to reference mitochondrion", fontsize=14, fontweight="demi")
        plt.xlabel("L2 distance")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.source_pipeline.output_dir}/combined_distances.png", dpi=200)

    def closest(self, n: int = 10):
        """Return the n closest mitos in the target dataset to the reference.

        Parameters:
            n: Number of entries to return.
        """
        return self.distances_df.head(n)

    def furthest(self, n: int = 10):
        """Return the n furthest mitos in the target dataset from the reference.

        Parameters:
            n: Number of entries to return.
        """
        return self.distances_df.dropna().tail(n)

    def closest_combined(self, n: int = 20):
        """Return the n closest mitos from the combined population."""
        return self.distances_combined_df.head(n)

    def furthest_combined(self, n: int = 20):
        """Return the n furthest mitos from the combined population."""
        return self.distances_combined_df.dropna().tail(n)

    def plot_closest(self, k: int = 10):
        self.plot_series(self.closest(k), title=f"{k} nearest in {self.target_pipeline.name}")

    def plot_furthest(self, k: int = 10):
        self.plot_series(self.furthest(k), title=f"{k} furthest in {self.target_pipeline.name}")

    def plot_closest_combined(self, k: int = 20):
        self.plot_series_combined(self.closest_combined(k), title=f"{k} nearest (combined)")
        plt.savefig(f"{self.source_pipeline.output_dir}/combined_closest_{k}.png", dpi=200)

    def plot_furthest_combined(self, k: int = 20):
        self.plot_series_combined(self.furthest_combined(k), title=f"{k} furthest (combined)")
        plt.savefig(f"{self.source_pipeline.output_dir}/combined_furthest_{k}.png", dpi=200)

    def plot_series(self, entries, pad_size: int = 200, title: str = ""):
        """Plot a row of representative mito thumbnails from the target dataset.

        Parameters:
            entries: DataFrame slice (from closest() or furthest()) iterated by mito_id.
            pad_size: Target height and width in pixels for each thumbnail.
            title: Optional suptitle for the figure.
        """
        import matplotlib.pyplot as plt
        from .math_helpers import pad_slice_to_size

        n = len(entries)
        fig, axs = plt.subplots(1, n, figsize=(11, 2))
        axs = np.array(axs).flatten()

        for i, (plot_mito_id, _) in enumerate(entries.iterrows()):
            ax = axs[i]
            entry = self.target_pipeline.mito_catalog[plot_mito_id]
            slc = entry.to_slice()
            try:
                slc = pad_slice_to_size(slc, pad_size, pad_size)
            except ValueError:
                pass
            z_index = slc.z.start
            self.vis_target.plot_mito_mask(slc, z_index=z_index, highlight_mito_id=plot_mito_id, ax=ax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel("")
            ax.set_xlabel("")
            ax.set_title(f"{i}: {plot_mito_id}", fontsize=7)

        plt.suptitle(title, fontsize=14, fontweight="demi")
        plt.tight_layout()
        plt.show()

    def plot_series_combined(self, entries, pad_size: int = 200, title: str = ""):
        """Plot a row of mito thumbnails from the combined population.

        Axes spines are colored by origin dataset: source color for source_pipeline
        mitos, target color for target_pipeline mitos.

        Parameters:
            entries: DataFrame slice from closest_combined() or furthest_combined(),
                must have an 'origin' column.
            pad_size: Target height and width in pixels for each thumbnail.
            title: Optional suptitle for the figure.
        """
        from .math_helpers import pad_slice_to_size

        n = len(entries)

        if n > 10:
            fig, axs = plt.subplots(2, n//2, figsize=(1.1 * n//2, 4))
        else:
            fig, axs = plt.subplots(1, n, figsize=(1.1 * n, 2))

        axs = np.array(axs).flatten()

        for i, (plot_mito_id, row) in enumerate(entries.iterrows()):
            ax = axs[i]
            origin = row["origin"]
            is_source = origin == self.source_pipeline.name
            pipeline = self.source_pipeline if is_source else self.target_pipeline
            vis = self.vis_source if is_source else self.vis_target
            spine_color = self.color_source if is_source else self.color_target

            entry = pipeline.mito_catalog[plot_mito_id]
            slc = entry.to_slice()
            spine_width = 3
            try:
                slc = pad_slice_to_size(slc, pad_size, pad_size)
            except ValueError:
                slc = pad_slice_to_size(slc, pad_size*2, pad_size*2)
                spine_width = 6
                pass

            vis.plot_mito_mask(slc, z_index=slc.z.start, highlight_mito_id=plot_mito_id, ax=ax)

            for spine in ax.spines.values():
                spine.set_edgecolor(spine_color)
                spine.set_linewidth(spine_width)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel("")
            ax.set_xlabel("")
            ax.set_title(f"{i}: {plot_mito_id}", fontsize=7)

        plt.suptitle(title, fontsize=14, fontweight="demi")
        plt.tight_layout()
        plt.show()

    def run(self, reference_mito_id_index: int = 0):
        """Run the full cross-dataset analysis for a single reference mito.

        Parameters:
            reference_mito_id_index: Index into source_pipeline.mito_ids() to use
                as the reference.
        """

        self.set_reference(self.source_pipeline.reference_analyzer.reference_mito_id)
        self.compute_distances()
        self.compute_combined_distances()
        self.plot_distance_distribution()
        self.plot_reference()
        self.plot_closest()
        self.plot_furthest()
        self.plot_closest_combined()
        self.plot_furthest_combined()
