"""Find the pool organs most like one organ in this scan (P5)."""


def _contrast_phase(ct_nii, stats):
    """Phase from the classifier, reusing the statistics already computed."""
    from totalsegmentator.bin.totalseg_get_phase import get_ct_contrast_phase

    final = None
    for item in get_ct_contrast_phase(ct_nii, device="cpu", quiet=True, existing_stats=stats):
        final = item
    return final["result"]["phase"]


def run(request, progress):
    import tempfile
    from pathlib import Path

    import nibabel as nib
    import numpy as np
    import SimpleITK as sitk

    from ..describe import describe
    from ..dicomweb import client_for, load_series
    from ..nifti import sitk_like, volume_to_sitk
    from ..similar import case_for_study, search
    from ..totalseg import run as totalseg, structure_names

    organ = request["organ"]
    client = client_for(request["dicomweb_url"])

    progress("fetching")
    volume = load_series(client, request["study_uid"], request["series_uid"])

    # Same arguments the phase classifier uses, so its stats can be reused.
    progress("segmenting")
    labelmap, stats = totalseg(
        volume,
        fast=True,
        statistics=True,
        stats_aggregation="median",
        statistics_exclude_masks_at_border=False,
    )

    values = [v for v, name in structure_names().items() if name == organ]
    if not values:
        raise ValueError(f"unknown organ: {organ}")
    mask = (labelmap == values[0]).astype(np.uint8)
    if not mask.any():
        raise ValueError(f"TotalSegmentator found no {organ} in this series")

    # Through a .nii.gz and nib.load, the same loader the index used.
    with tempfile.TemporaryDirectory(prefix="argus-similar-") as workdir:
        ct_path = Path(workdir) / "ct.nii.gz"
        mask_path = Path(workdir) / "mask.nii.gz"
        ct_image = volume_to_sitk(volume)
        sitk.WriteImage(ct_image, str(ct_path))
        sitk.WriteImage(sitk_like(ct_image, mask), str(mask_path))
        ct_nii = nib.load(ct_path)
        mask_nii = nib.load(mask_path)

        progress("classifying")
        phase = _contrast_phase(ct_nii, stats)

        progress("describing")
        baseline, embedding = describe(ct_nii, mask_nii)

    if baseline is None or embedding is None:
        raise ValueError(f"{organ} too small to describe")

    progress("searching")
    results = search(
        baseline,
        embedding,
        organ=organ,
        phase=phase,
        k=request.get("k", 5),
        phase_centred=request.get("phase_centred", False),
        exclude_case=request.get("exclude_case") or case_for_study(request["study_uid"]),
    )
    return {"organ": organ, "phase": phase, "results": results}
