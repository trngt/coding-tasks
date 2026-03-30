from .slice import Slice
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

    def plot_segmentation_example(self):

        # Define a tile region
        y_start, y_end = 64, 192
        x_start, x_end = 64, 192
        z_start, z_end = 64, 192

        y_start, y_end = 90, 140
        x_start, x_end = 90, 140
        z_start, z_end = 90, 140

        # A slice on the z-axis
        z_mid_em = 130#(z_start+z_end)//2

        # Load segmentation volume
        seg_vol = self.data_manager.segmentation_data.data.isel(
            z=slice(z_start, z_end),
            y=slice(y_start, y_end),
            x=slice(x_start, x_end)
        ).compute().values  # shape: (Z, Y, X)

        data = self.data_manager.em_data.data

        # Get the physical coordinate bounds of the EM tile
        y_min = float(data.y[y_start])
        y_max = float(data.y[y_end - 1])
        x_min = float(data.x[x_start])
        x_max = float(data.x[x_end - 1])
        z_nm = float(data.z[z_mid_em])

        print(f"Physical bounds: z={z_nm:.1f}, y={y_min:.1f}–{y_max:.1f}, x={x_min:.1f}–{x_max:.1f} nm")

        # Select EM tile
        em_slice = self.data_manager.em_data.data.isel(
            z=z_mid_em,
            y=slice(y_start, y_end),
            x=slice(x_start, x_end)
        ).compute()

        # Select z by physical coordinate (nearest), then y/x by physical range (slice)
        seg_slice = self.data_manager.segmentation_data.data.isel(z=z_mid_em,
            y=slice(y_start, y_end),
            x=slice(x_start, x_end)
        ).compute()

        fig, axes = plt.subplots(1, 3, figsize=(10, 3))

        extent = [0, 1000, 0, 1000]

        axes[0].imshow(em_slice, cmap='gray', extent=extent)
        axes[0].set_title('EM')

        cmap, norm, clim, seg_out = self._build_seg_colormap(seg_slice.values, remap=True)

        masked_slice = np.ma.masked_where(seg_slice == 0, seg_slice)
        axes[1].imshow(seg_out, cmap=cmap, norm=norm, extent=extent)
        axes[1].set_title('Segmentation')

        # Overlay
        axes[2].imshow(em_slice, cmap='gray', extent=extent)
        axes[2].imshow(seg_out, cmap=cmap, norm=norm, extent=extent, alpha=0.4)
        axes[2].set_title('Overlay')

        plt.tight_layout()
        plt.show()

    def plot_segmentation_3d_voxels(self):
        """Renders the segmentation volume as coloured voxels using PyVista."""

        import numpy as np
        import pyvista as pv
        from matplotlib import colormaps

        # Define tile region
        y_start, y_end = 64, 192
        x_start, x_end = 64, 192
        z_start, z_end = 64, 192

        y_start, y_end = 90, 140
        x_start, x_end = 90, 140
        z_start, z_end = 90, 140

        # Load segmentation volume
        seg_vol = self.data_manager.segmentation_data.data.isel(
            z=slice(z_start, z_end),
            y=slice(y_start, y_end),
            x=slice(x_start, x_end)
        ).compute().values  # shape: (Z, Y, X)

        # Mask out background (id=0)
        mask = seg_vol > 0

        # Build a PyVista UniformGrid matching the voxel array
        nz, ny, nx = seg_vol.shape
        grid = pv.ImageData()
        grid.dimensions = (nx + 1, ny + 1, nz + 1)  # cell-corner dimensions
        grid.spacing = (1, 1, 1)
        grid.origin = (0, 0, 0)

        cmap, norm, clim, seg_out = self._build_seg_colormap(seg_vol, remap=True)

        # Attach segment IDs as cell data (flattened in Fortran/ZYX order)
        grid.cell_data["segment_id"] = seg_out.filled(0).flatten(order="C").astype(np.float32)

        # Threshold to remove background cells
        thresholded = grid.threshold(value=0.5, scalars="segment_id")

        # Plot
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

        # Plot the z surface corresponding to the
        # 2D plots
        z_mid = (130 - z_start)
        plane = pv.Plane(
            center=(nx / 2, ny / 2, z_mid),
            direction=(0, 0, 1),
            i_size=nx,
            j_size=ny,
        )

        plotter.add_mesh(
            plane,
            color="white",
            opacity=0.4,
            show_edges=False,
        )

        plotter.show(title="Segmentation — voxel render", jupyter_backend="static")
