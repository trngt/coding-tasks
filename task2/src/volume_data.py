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
        raise NotImplementedError
