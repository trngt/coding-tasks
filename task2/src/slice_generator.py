from typing import List
from .slice import Slice


class SliceGenerator:
    """Produces a predefined list of Slice objects.

    Decoupled from DataManager.
    """

    def __init__(self):
        pass

    def generate(self) -> List[Slice]:
        raise NotImplementedError
