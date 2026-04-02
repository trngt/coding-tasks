from .slice import Slice3D
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

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

    def plot_segmentation_example(self, slc: Slice3D, z_index: int, title: str = ""):
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

        plt.suptitle(title)

        plt.tight_layout()
        plt.show()

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

def format_microscopy_ax(ax, data_manager, slc):

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
            ax.axvline(x, color='red', linewidth=0.25, alpha=0.1)
        for y in y_grid_lines:
            ax.axhline(y, color='red', linewidth=0.25, alpha=0.1)

    format_axes(ax)
    draw_grid(ax)