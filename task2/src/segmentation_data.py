from .volume_data import VolumeData


class SegmentationData(VolumeData):
    """Loads segmentation label data from S3.

    Handles resolution selection with a fixed +1 offset relative to EM.
    dtype: uint32
    """

    def __init__(self, s3_path: str, resolution: str = "s0"):
        super().__init__(s3_path, resolution)
