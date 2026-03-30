from typing import List, Tuple
import numpy as np
from .slice import Slice
from .em_data import EMData
from .segmentation_data import SegmentationData


class DataManager:
    """Data manager for EM and segmentation data. 

    Dictates loading and defining slices of the data for training.
    """

    def __init__(self, group_url: str, segmentation_url: str, resolution: str ='s3'):

        # Initialize the EM dataset
        self.em_data = EMData(group_url, resolution)

        # Initialize the segmentation dataset
        # Decrement segmentation resolution by 1 (em data: s1, segmentation data: s0)
        # todo: With the data I've worked with thus far, the segmentation data
        # as half the resolution of the em data.
        segmentation_resolution = 's{}'.format(int(resolution.replace('s', ''))-1)
        self.segmentation_data = SegmentationData(segmentation_url, segmentation_resolution)

        # Load the data for each
        self.em_data.load()
        self.segmentation_data.load()
