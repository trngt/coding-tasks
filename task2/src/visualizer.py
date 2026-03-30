from .slice import Slice


class Visualizer:
    """Plots EM slices, segmentation slices, and overlays.

    Separate from DataManager — no data ownership.
    Useful for debugging and validation.
    """

    def __init__(self, data_manager):
        self.data_manager = data_manager

    def plot_segmentation_example(self):

        # Define a tile region
        y_start, y_end = 128, 256
        x_start, x_end = 128, 256

        data = self.data_manager.em_data.data

        z_mid_em = data.shape[2]//2

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


        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import numpy as np

        fig, axes = plt.subplots(1, 3, figsize=(10, 3))

        extent = [0, 1000, 0, 1000]

        axes[0].imshow(em_slice, cmap='gray', extent=extent)
        axes[0].set_title('EM')

        axes[1].imshow(seg_slice, cmap='tab20', extent=extent)
        axes[1].set_title('Segmentation')

        # Overlay
        axes[2].imshow(em_slice, cmap='gray', extent=extent)
        axes[2].imshow(seg_slice, cmap='tab20', alpha=0.4, extent=extent)
        axes[2].set_title('Overlay')

        plt.tight_layout()
        plt.show()
