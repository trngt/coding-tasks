from typing import List, Tuple
import numpy as np
from .em_data import EMData
from .segmentation_data import SegmentationData


class DataManager:
    """Data manager for EM and segmentation data. 

    Dictates loading and defining slices of the data for training.
    """

    def __init__(self, group_url: str, segmentation_url: str, em_resolution: str,
            segmentation_resolution: str, name: str):

        # Initialize the EM dataset
        self.em_data = EMData(group_url, em_resolution)

        # Initialize the segmentation dataset
        self.segmentation_data = SegmentationData(segmentation_url, segmentation_resolution)

        # Load the data for each
        self.em_data.load()
        self.segmentation_data.load()
        self.name = name
