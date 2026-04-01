import numpy as np
from .slice import Slice3D
from fibsem_tools import read_xarray, read

class VolumeData:
    """Base class for loading volumetric data from S3-hosted N5/Zarr datasets."""

    def __init__(self, s3_path: str, resolution: str = "s0"):
        self.creds = {'anon': True}  # anonymous credentials for Amazon S3
        self.s3_path = s3_path
        self.resolution = resolution

    def view_groups(self):
        self.group = read(self.s3_path, storage_options=self.creds).arrays()
        from rich import print  # pretty printing
        print(tuple(self.group))

    def load(self):
        self.data = read_xarray(self.s3_path + f'/{self.resolution}', storage_options=self.creds)

    def get_slice(self, slc: Slice3D) -> np.ndarray:
        """Get the slice of the data as an np.array"""

        # If selecting a plane on the z-axis, modify the argument
        # to a single int rather than a slice/range
        if slc.z.start == slc.z.stop:
            z = slc.z.start
        else:
            z = slc.z

        slice_data = self.data.isel(
            z=z,
            y=slc.y,
            x=slc.x
        )
        slice_data = np.array(slice_data.values)

        return slice_data
