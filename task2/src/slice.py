class Slice:
    """A simple 3D index range definition (z, y, x ranges).

    No knowledge of the data itself.
    """

    def __init__(self, z: slice, y: slice, x: slice):
        self.z = z
        self.y = y
        self.x = x
