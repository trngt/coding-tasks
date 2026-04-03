from .slice import Slice3D


def pad_slice_to_size(slc: Slice3D, target_h: int, target_w: int) -> Slice3D:
    """Expand a Slice3D to a target height and width by adding padding.

    Padding is distributed evenly on both sides of each axis. If the
    required padding is odd, the extra pixel is added to the end (right/bottom).
    The z dimension is unchanged.

    Parameters:
        slc: Input Slice3D to pad.
        target_h: Desired height (y extent) in pixels.
        target_w: Desired width (x extent) in pixels.

    Returns:
        A new Slice3D with the padded extents.

    Raises:
        ValueError: If the slice is already larger than the target on either axis.
    """
    current_h = slc.y.stop - slc.y.start
    current_w = slc.x.stop - slc.x.start

    if current_h > target_h:
        raise ValueError(f"Slice height {current_h} exceeds target {target_h}")
    if current_w > target_w:
        raise ValueError(f"Slice width {current_w} exceeds target {target_w}")

    pad_h = target_h - current_h
    pad_w = target_w - current_w

    y_start = slc.y.start - pad_h // 2
    y_stop  = slc.y.stop  + (pad_h - pad_h // 2)

    x_start = slc.x.start - pad_w // 2
    x_stop  = slc.x.stop  + (pad_w - pad_w // 2)

    return Slice3D(
        z=slc.z,
        y=slice(y_start, y_stop),
        x=slice(x_start, x_stop),
    )
