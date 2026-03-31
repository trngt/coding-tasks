from typing import List
from src.slice import Slice3D
from .data_manager import DataManager

class SliceGenerator:
    """Produces a predefined list of Slice objects.

    Decoupled from DataManager.
    """

    def __init__(self, data_manager: DataManager, patch_size, z_step):

        self.patch_size = patch_size
        self.z_step = z_step
        self.data_manager = data_manager


    def generate(self) -> List[Slice3D]:

        from itertools import product

        def tile_slices(size, tile_size):
            """Create the tiles slices along a dimension, 
            given the total size and the patch size"""
            return [slice(i, min(i + tile_size, size)) for i in range(0, size, tile_size)]

        # Volume dimensions
        Z, Y, X = self.data_manager.em_data.data.shape

        # Create the tiles for the X and Y dimension
        # The Z slice will be a single index
        x_slices = tile_slices(X, self.patch_size)
        y_slices = tile_slices(Y, self.patch_size)
        z_indices = range(0, Z, self.z_step)  # each z is a single plane

        self.patches = []
        for z, ys, xs in product(z_indices, y_slices, x_slices):
            patch_definition = Slice3D(slice(z, z), ys, xs)
            self.patches.append(patch_definition)

        return self.patches
            
