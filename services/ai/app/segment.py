import numpy as np


def threshold_segment(volume, low, high):
    array = volume.array

    if array.ndim != 3:
        raise ValueError(
            f"Expected a 3D volume, got shape {array.shape}"
        )

    return (
        (array >= low) &
        (array <= high)
    ).astype(np.uint8)


def calculate_bounding_box(mask):
    z, row, col = np.nonzero(mask)

    if len(z) == 0:
        return None

    return {
        "z_min": int(z.min()),
        "z_max": int(z.max()),
        "row_min": int(row.min()),
        "row_max": int(row.max()),
        "col_min": int(col.min()),
        "col_max": int(col.max()),
    }

