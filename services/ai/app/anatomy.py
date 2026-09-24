"""TotalSegmentator structure names to SNOMED codes.

A segment label is free text; `segmented_property_type` is a coded concept,
and it is what another system actually reads. "liver" as a string means
nothing to anyone downstream.

This covers the structures worth naming properly and falls back to a generic
anatomical structure for the rest. Doing all 117 by hand before anything works
would be the wrong order — but a wrong code is worse than a generic one, so
nothing here is a guess.
"""

from __future__ import annotations

from pydicom.sr.codedict import codes

FALLBACK = codes.SCT.AnatomicalStructure

# TotalSegmentator name -> SNOMED concept. Laterality is dropped: a left and a
# right kidney are both a kidney, and the segment label keeps the side.
_EXACT = {
    "liver": codes.SCT.Liver,
    "spleen": codes.SCT.Spleen,
    "pancreas": codes.SCT.Pancreas,
    "stomach": codes.SCT.Stomach,
    "gallbladder": codes.SCT.Gallbladder,
    "duodenum": codes.SCT.Duodenum,
    "colon": codes.SCT.Colon,
    "small_bowel": codes.SCT.SmallIntestine,
    "esophagus": codes.SCT.Esophagus,
    "trachea": codes.SCT.Trachea,
    "aorta": codes.SCT.Aorta,
    "heart": codes.SCT.Heart,
    "thyroid_gland": codes.SCT.Thyroid,
    "prostate": codes.SCT.Prostate,
    "brain": codes.SCT.Brain,
    "sternum": codes.SCT.Sternum,
}

# Prefix -> concept, for the families TotalSegmentator numbers or sides.
_PREFIXES = (
    ("kidney", codes.SCT.Kidney),
    ("lung_", codes.SCT.Lung),
    ("vertebrae_", codes.SCT.Vertebra),
    ("rib_", codes.SCT.Rib),
)


def concept_for(name: str):
    """The coded concept for a TotalSegmentator structure name."""
    key = name.lower()

    if key in _EXACT:
        return _EXACT[key]

    for prefix, concept in _PREFIXES:
        if key.startswith(prefix):
            return concept

    return FALLBACK


def label_for(name: str) -> str:
    """A readable segment label — 'kidney left', not 'kidney_left'."""
    return name.replace("_", " ")
