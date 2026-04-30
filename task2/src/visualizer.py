from .slice import Slice3D
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.cm as cm
from sklearn.manifold import TSNE


class Visualizer:
    """Plots EM slices, segmentation slices, and overlays.

    Separate from DataManager — no data ownership.
    Useful for debugging and validation.
    """

    def __init__(self, data_manager):
        self.data_manager = data_manager

    def _build_seg_colormap(self, seg_vol, remap=False):
        """Returns a consistent colormap for both the
        2D (Matplotlib) and 3D (PyVista) slices of the volume data.
        Remap applies to the 2D plots.
        """
        unique_ids = np.unique(seg_vol)
        unique_ids = unique_ids[unique_ids > 0]  # drop background

        base_colors = plt.cm.tab20(np.linspace(0, 1, 20))
        colors = [tuple(base_colors[int(i) % len(base_colors)]) for i in range(len(unique_ids))]

        cmap = mcolors.ListedColormap(colors)
        cmap.set_bad(alpha=0)

        # To enforce the 2d and 3d plots are consistently colored,
        # we need to remap the unique ids to the same color map, and
        # return the updated segmentation volume
        if remap:
            id_to_index = {seg_id: i + 1 for i, seg_id in enumerate(unique_ids)}
            seg_out = np.zeros_like(seg_vol, dtype=float)
            for seg_id, idx in id_to_index.items():
                seg_out[seg_vol == seg_id] = idx
            seg_out = np.ma.masked_where(seg_vol == 0, seg_out)
            bounds = np.arange(len(unique_ids) + 2) - 0.5
            norm = mcolors.BoundaryNorm(bounds, ncolors=(len(unique_ids)+1))
            clim = [0.5, len(unique_ids) + 0.5]
        else:
            seg_out = np.ma.masked_where(seg_vol == 0, seg_vol.astype(float))
            bounds = np.concatenate([unique_ids - 0.5, [unique_ids[-1] + 0.5]])
            norm = mcolors.BoundaryNorm(bounds, ncolors=(len(unique_ids)+1))
            clim = [bounds[0], bounds[-1]]

        return cmap, norm, clim, seg_out

    def plot_segmentation_example(self, slc: Slice3D, z_index: int, title: str = None,
        highlight_mito_id: int = None):
        """Plot EM, segmentation, and overlay for a given slice at z_index.

        Args:
            slc: Slice3D object defining the (z, y, x) index ranges to load.
            z_index: Index along the z axis (within slc.z bounds) to display.
        """
        em_slice = self.data_manager.em_data.data.isel(
            z=z_index,
            y=slc.y,
            x=slc.x,
        ).compute()

        seg_slice = self.data_manager.segmentation_data.data.isel(
            z=z_index,
            y=slc.y,
            x=slc.x,
        ).compute()

        if highlight_mito_id is not None:
            seg_slice = seg_slice * (seg_slice == highlight_mito_id)

        cmap, norm, clim, seg_out = self._build_seg_colormap(seg_slice.values, remap=True)

        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(9, 3.5))

        # Compute extents in nm
        dimensions = self.data_manager.em_data.data.attrs['pixelResolution']['dimensions']
        x_nm_per_pixel, y_nm_per_pixel, z_nm_per_pixel = tuple(dimensions)
        nm_per_microns = 1000

        extent = compute_extents(self.data_manager, slc)

        ax0.imshow(em_slice, cmap='gray', extent=extent)
        ax0.set_title('EM')
        format_microscopy_ax(ax0, self.data_manager, slc)

        ax1.imshow(seg_out, cmap=cmap, norm=norm, extent=extent)
        ax1.set_title('Segmentation')
        format_microscopy_ax(ax1, self.data_manager, slc)

        ax2.imshow(em_slice, cmap='gray', extent=extent)
        ax2.imshow(seg_out, cmap=cmap, norm=norm, alpha=0.4, extent=extent)
        ax2.set_title('Overlay')
        format_microscopy_ax(ax2, self.data_manager, slc)

        if title is None:
            title = f"z:{slc.z.start}, y:{slc.y.start}-{slc.y.stop}, x:{slc.x.start}-{slc.x.stop}"

        plt.suptitle(title)

        plt.tight_layout()
        plt.show()

    def plot_mito_mask(self, slc: Slice3D, z_index: int, title: str = None,
            highlight_mito_id: int = None, alpha: float = 0.75, ax=None):
        """Plot EM data masked by segmentation, highlighting mitochondria.
        """
        em_slice = self.data_manager.em_data.data.isel(
            z=z_index,
            y=slc.y,
            x=slc.x,
        ).compute()

        seg_slice = self.data_manager.segmentation_data.data.isel(
            z=z_index,
            y=slc.y,
            x=slc.x,
        ).compute()

        # Build boolean mask: True where there is no mitochondria (to be blacked out)
        if highlight_mito_id is not None:
            non_mito_mask = seg_slice.values != highlight_mito_id
        else:
            non_mito_mask = seg_slice.values == 0

        # RGBA mask: black + opaque where non-mito, fully transparent where mito
        mask_rgba = np.zeros((*non_mito_mask.shape, 4), dtype=np.float32)
        mask_rgba[non_mito_mask] = [0, 0, 0, alpha]   # black, semi-opaque
        mask_rgba[~non_mito_mask] = [0, 0, 0, 0]       # transparent

        extent = compute_extents(self.data_manager, slc)

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))

        ax.imshow(em_slice, cmap='gray', extent=extent)
        ax.imshow(mask_rgba, extent=extent)

        if title is None:
            title = f"z:{slc.z.start}, y:{slc.y.start}-{slc.y.stop}, x:{slc.x.start}-{slc.x.stop}"

        ax.set_title(title)
        format_microscopy_ax(ax, self.data_manager, slc)


    def plot_segmentation_3d_voxels(self, slc: Slice3D, z_plane: int):
        """Renders the segmentation volume as coloured voxels using PyVista.

        todo: Currently unused

        Args:
            slc: Slice3D object defining the (z, y, x) index ranges to load.
            z_plane: Z index of the reference plane to overlay (absolute index).
        """
        import pyvista as pv

        seg_vol = self.data_manager.segmentation_data.data.isel(
            z=slc.z,
            y=slc.y,
            x=slc.x,
        ).compute().values  # shape: (Z, Y, X)

        mask = seg_vol > 0

        nz, ny, nx = seg_vol.shape
        grid = pv.ImageData()
        grid.dimensions = (nx + 1, ny + 1, nz + 1)
        grid.spacing = (1, 1, 1)
        grid.origin = (0, 0, 0)

        cmap, norm, clim, seg_out = self._build_seg_colormap(seg_vol, remap=True)

        grid.cell_data["segment_id"] = seg_out.filled(0).flatten(order="C").astype(np.float32)

        thresholded = grid.threshold(value=0.5, scalars="segment_id")

        plotter = pv.Plotter(window_size=[300, 300])
        plotter.add_mesh(
            thresholded,
            scalars="segment_id",
            cmap=cmap,
            clim=[norm.boundaries[0], norm.boundaries[-1]],
            show_edges=False,
            opacity=1.0,
        )
        plotter.remove_scalar_bar()

        z_local = z_plane - slc.z.start
        plane = pv.Plane(
            center=(nx / 2, ny / 2, z_local),
            direction=(0, 0, 1),
            i_size=nx,
            j_size=ny,
        )

        plotter.add_mesh(plane, color="white", opacity=0.4, show_edges=False)
        plotter.show(title="Segmentation — voxel render", jupyter_backend="static")

def compute_extents(data_manager, slc):
    dimensions = data_manager.em_data.data.attrs['pixelResolution']['dimensions']
    x_nm_per_pixel, y_nm_per_pixel, z_nm_per_pixel = tuple(dimensions)
    nm_per_microns = 1000

    extent = [slc.x.start*x_nm_per_pixel/nm_per_microns, 
                  slc.x.stop*x_nm_per_pixel/nm_per_microns,
                  slc.y.start*y_nm_per_pixel/nm_per_microns, 
                  slc.y.stop*y_nm_per_pixel/nm_per_microns]
    return extent

def format_microscopy_ax(ax, data_manager, slc, grid_alpha=0.25):

    # Compute extents in nm
    dimensions = data_manager.em_data.data.attrs['pixelResolution']['dimensions']
    x_nm_per_pixel, y_nm_per_pixel, z_nm_per_pixel = tuple(dimensions)
    nm_per_microns = 1000

    extent = compute_extents(data_manager, slc)

    def format_axes(ax):
        """Format the axes for each imshow with the same labels"""
        ax.set_xlabel(f"{extent[1]-extent[0]:.1f} μm, {slc.x.stop-slc.x.start} px")
        ax.set_ylabel(f"{extent[3]-extent[2]:.1f} μm, {slc.y.stop-slc.y.start} px")

    # Draw the grid
    patch_size = 16 # todo: fixed based on DINOv3 patch size
    x_patch_um = patch_size * x_nm_per_pixel / nm_per_microns
    y_patch_um = patch_size * y_nm_per_pixel / nm_per_microns
    x_grid_lines = np.arange(extent[0], extent[1], x_patch_um)
    y_grid_lines = np.arange(extent[2], extent[3], y_patch_um)
    def draw_grid(ax):
        for x in x_grid_lines:
            ax.axvline(x, color='red', linewidth=0.25, alpha=grid_alpha)
        for y in y_grid_lines:
            ax.axhline(y, color='red', linewidth=0.25, alpha=grid_alpha)

    format_axes(ax)
    draw_grid(ax)


def create_cosine_distance_matrix(embeddings):
    """
    Compute and plot a pairwise cosine distance matrix as a heatmap.
    """
    embeddings = np.asarray(embeddings, dtype=float)
    n = embeddings.shape[0]

    # ── 1. Normalise rows → cosine distance = 1 - dot product ─────────────
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)          # guard against zero vectors
    embeddings_normed = embeddings/norms
    dist_matrix = cdist(embeddings_normed, embeddings_normed, metric='cosine')
    np.fill_diagonal(dist_matrix, 0.0)              # fix any floating-point noise on diagonal
    return dist_matrix


def plot_distance_matrix(dist_matrix, labels=None, cmap='Reds',
    title='Mitochondria Cosine Distance Matrix', figsize=(5, 4)):
    # ── 2. Build figure ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(dist_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Cosine Distance  (0 = identical, 1 = orthogonal)', fontsize=10)

    # ── 3. Tick labels ─────────────────────────────────────────────────────
    if labels is not None:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
    else:
        ax.set_xlabel('Mitochondrion Index', fontsize=11)
        ax.set_ylabel('Mitochondrion Index', fontsize=11)

    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.show()

    return dist_matrix, fig, ax


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize


def plot_tsne(embeddings, labels=None, color_by=None, perplexity=30, n_iter=1000,
              metric='cosine', title='Mitochondria t-SNE Projection',
              figsize=(5, 4), cmap='plasma', point_size=30, random_state=42):
    """
    Reduce embeddings to 2-D with t-SNE and plot the projection.
    """
    embeddings = np.asarray(embeddings, dtype=float)
    n = len(embeddings)

    # ── 1. Fit t-SNE ───────────────────────────────────────────────────────
    if metric == 'cosine':
        # Pass a precomputed cosine distance matrix for accuracy
        normed = normalize(embeddings, norm='l2')
        dist_matrix = 1.0 - (normed @ normed.T)          # shape (n, n)
        np.fill_diagonal(dist_matrix, 0.0)
        np.clip(dist_matrix, 0, None, out=dist_matrix)   # guard float negatives

        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, n - 1),
            # n_iter=n_iter,
            metric='precomputed',
            random_state=random_state,
            init='random',                # required when metric='precomputed'
        )
        embedding_2d = tsne.fit_transform(dist_matrix)
    else:
        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, n - 1),
            n_iter=n_iter,
            metric=metric,
            random_state=random_state,
            init='pca',
        )
        embedding_2d = tsne.fit_transform(embeddings)

    # ── 2. Plot ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)

    if labels is not None:
        unique_labels = list(dict.fromkeys(labels))
        palette = cm.get_cmap(cmap, len(unique_labels))
        label_to_color = {lbl: palette(i) for i, lbl in enumerate(unique_labels)}

        for lbl in unique_labels:
            mask = np.array(labels) == lbl
            ax.scatter(
                embedding_2d[mask, 0], embedding_2d[mask, 1],
                c=[label_to_color[lbl]], s=point_size,
                label=lbl, alpha=0.85, edgecolors='none',
            )
        ax.legend(title='Group', bbox_to_anchor=(1.02, 1), loc='upper left',
                  framealpha=0.3, fontsize=8)

    elif color_by is not None:
        color_by = np.asarray(color_by, dtype=float)
        sc = ax.scatter(
            embedding_2d[:, 0], embedding_2d[:, 1],
            c=color_by, cmap=cmap, s=point_size, alpha=0.85, edgecolors='none',
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Value', fontsize=10)

    else:
        ax.scatter(
            embedding_2d[:, 0], embedding_2d[:, 1],
            c=np.arange(n), cmap=cmap, s=point_size, alpha=0.85, edgecolors='none',
        )

    ax.set_xlabel('t-SNE 1', fontsize=11)
    ax.set_ylabel('t-SNE 2', fontsize=11)
    ax.set_title(f'{title}\n(perplexity={perplexity}, n_iter={n_iter})', fontsize=12,
                 fontweight='bold', pad=12)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()

    return embedding_2d, fig, ax
