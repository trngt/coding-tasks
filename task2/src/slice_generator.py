from typing import List
from src.slice import Slice3D
from .data_manager import DataManager

class SliceGenerator:
    """Produces a predefined list of Slice3D from a 3D volume from the data manager.
    """

    def __init__(self, data_manager: DataManager, patch_size: int, z_step: int, 
        inset: int, tile_step: int = None):
        """
        Initialize the slice generator.

        Parameters:
            data_manager: of the 3D volume dataset
            patch_size: the x and y dimensions of each patch
            z_step: the amount of step along the z-dimension to create patches
            inset: the number of pixels to inset along the boundaries of the volume 
                (to remove edge effects)
            tile_step: the pixels to step between each patch, <patch_size will create overlapping
                patches within a plane. Defaults to patch_size//2
        """

        self.patch_size = patch_size
        self.z_step = z_step
        self.data_manager = data_manager
        self.inset = inset
        self.tile_step = patch_size//2 # Half-steps to handle mitochondria on edges


    def generate(self) -> List[Slice3D]:

        from itertools import product

        def tile_slices(size, tile_size, tile_step):
            """Create the tiles slices along a dimension, 
            given the total size and the patch size"""
            return [slice(i, min(i + tile_size, size)) 

                # End early, to create whole tiles
                for i in range(self.inset, size-self.inset-tile_step, tile_step)]

        # Volume dimensions
        Z, Y, X = self.data_manager.em_data.data.shape

        # Create the tiles for the X and Y dimension
        # The Z slice will be a single index
        x_slices = tile_slices(X, self.patch_size, self.tile_step)
        y_slices = tile_slices(Y, self.patch_size, self.tile_step)
        z_indices = range(self.inset, Z-self.inset, self.z_step)  # each z is a single plane

        self.patches = []
        for z, ys, xs in product(z_indices, y_slices, x_slices):
            patch_definition = Slice3D(slice(z, z), ys, xs)
            self.patches.append(patch_definition)

        return self.patches
            
