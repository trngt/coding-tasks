class Slice3D:
    """A simple 3D index range definition (z, y, x ranges).

    Note the x, y, and z arguments are slices (ranges of values)
    """

    def __init__(self, z: slice, y: slice, x: slice):
        self.z = z
        self.y = y
        self.x = x

    def size(self):
        return self.y.stop-self.y.start, self.x.stop-self.x.start

    def __repr__(self):
        return f"Slice at (z:{self.z}, y:{self.y}, x:{self.x})"

