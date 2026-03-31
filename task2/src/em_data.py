from .volume_data import VolumeData


class EMData(VolumeData):
    """Loads EM data from S3.
    """

    def __init__(self, s3_path: str, resolution: str = "s0"):
        super().__init__(s3_path, resolution)
