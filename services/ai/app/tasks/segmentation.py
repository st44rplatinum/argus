"""Threshold or TotalSegmentator, optionally stored back as a DICOM SEG."""


def run(request, progress):
    """Storing is the last action deliberately: terminating a cancelled job
    partway cannot then leave half a SEG in the PACS.
    """
    import numpy as np

    from ..dicom_seg import build_multi_segmentation, build_segmentation, store_segmentation
    from ..dicomweb import client_for, load_series
    from ..segment import calculate_bounding_box, threshold_segment

    client = client_for(request["dicomweb_url"])

    progress("fetching")
    volume = load_series(client, request["study_uid"], request["series_uid"])

    progress("inferring")
    if request.get("model", "threshold") == "totalsegmentator":
        from ..totalseg import run as totalseg, structure_names

        labelmap = totalseg(volume, fast=request.get("fast", True))
        names = structure_names()
        present = [int(v) for v in np.unique(labelmap) if v != 0]
        if not present:
            raise ValueError("TotalSegmentator found no structures")

        result = {
            "shape": list(volume.shape),
            "spacing_mm": list(volume.spacing),
            "voxel_count": int((labelmap != 0).sum()),
            "structures": [names.get(v, str(v)) for v in present],
            "seg_series_uid": None,
        }
        build = lambda sources: build_multi_segmentation(  # noqa: E731
            sources, labelmap, volume.sop_uids, names
        )
    else:
        mask = threshold_segment(volume, request["low"], request["high"])
        box = calculate_bounding_box(mask)
        if box is None:
            raise ValueError(f"No voxels in [{request['low']}, {request['high']}] HU")

        result = {
            "shape": list(volume.shape),
            "spacing_mm": list(volume.spacing),
            "voxel_count": int(mask.sum()),
            "bounding_box": box,
            "seg_series_uid": None,
        }
        build = lambda sources: build_segmentation(sources, mask, volume.sop_uids)  # noqa: E731

    if request.get("store"):
        progress("encoding")
        sources = client.retrieve_series(
            study_instance_uid=request["study_uid"],
            series_instance_uid=request["series_uid"],
        )
        segmentation = build(sources)

        progress("storing")
        result["seg_series_uid"] = store_segmentation(client, segmentation)

    return result
