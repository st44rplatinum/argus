"""Pull a working subset out of the TotalSegmentator zip, without unpacking it.

Extracting the whole archive costs about 45 GB on top of the 23.58 GB zip.
Almost none of that earns its space: a case is one CT plus 117 masks, and any
given experiment wants a handful of structures.

So this reads members out of the zip directly and writes only what was asked
for, then records what it took in a manifest. The manifest is the point — it
makes the subset reproducible, so the zip can be deleted and the same subset
rebuilt later from a fresh download.

  python scripts/subset_dataset.py --list
  python scripts/subset_dataset.py --cases 100 --structures liver spleen
"""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import defaultdict
from pathlib import Path

ZIP = Path("data/raw/Totalsegmentator_dataset_v201.zip")
OUT = Path("data/raw/totalsegmentator")
MANIFEST = OUT / "manifest.json"

CT_NAME = "ct.nii.gz"


def index_cases(archive: zipfile.ZipFile) -> dict[str, dict]:
    """Case id -> {'ct': member, 'structures': {name: member}}."""
    cases: dict[str, dict] = defaultdict(lambda: {"ct": None, "structures": {}})

    for info in archive.infolist():
        if info.is_dir():
            continue

        parts = Path(info.filename).parts
        if len(parts) < 2:
            continue

        case_id = parts[0]

        if parts[-1] == CT_NAME:
            cases[case_id]["ct"] = info
        elif "segmentations" in parts and parts[-1].endswith(".nii.gz"):
            name = parts[-1][: -len(".nii.gz")]
            cases[case_id]["structures"][name] = info

    # Cases with no CT are not usable.
    return {k: v for k, v in cases.items() if v["ct"] is not None}


def human(size: int) -> str:
    return f"{size / 1e9:.2f} GB" if size > 1e9 else f"{size / 1e6:.0f} MB"


def show(cases: dict[str, dict]) -> None:
    structures = sorted({name for c in cases.values() for name in c["structures"]})
    total_ct = sum(c["ct"].file_size for c in cases.values())

    print(f"cases           {len(cases)}")
    print(f"structures      {len(structures)}")
    print(f"CTs only        {human(total_ct)}")
    print(f"  per case      {human(total_ct // max(len(cases), 1))}")
    print()
    print("structures available:")
    for i in range(0, min(len(structures), 40), 4):
        print("   " + "  ".join(f"{s:24}" for s in structures[i : i + 4]))
    if len(structures) > 40:
        print(f"   ... and {len(structures) - 40} more")


def extract(
    archive: zipfile.ZipFile,
    cases: dict[str, dict],
    chosen: list[str],
    structures: list[str],
    out: Path,
) -> int:
    written = 0

    for case_id in chosen:
        case = cases[case_id]
        target = out / case_id
        target.mkdir(parents=True, exist_ok=True)

        wanted = [case["ct"]]
        for name in structures:
            member = case["structures"].get(name)
            if member is not None:
                wanted.append(member)

        for member in wanted:
            destination = target / Path(member.filename).name
            if Path(member.filename).parts[-2] == "segmentations":
                destination = target / "segmentations" / Path(member.filename).name
                destination.parent.mkdir(parents=True, exist_ok=True)

            with archive.open(member) as source, open(destination, "wb") as sink:
                while chunk := source.read(1 << 20):
                    sink.write(chunk)
                    written += len(chunk)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=ZIP)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--list", action="store_true", help="show contents and exit")
    parser.add_argument("--cases", type=int, default=50, help="how many cases to take")
    parser.add_argument(
        "--structures",
        nargs="*",
        default=["liver", "spleen", "kidney_left", "kidney_right"],
        help="structure masks to keep; empty keeps none, only the CTs",
    )
    args = parser.parse_args()

    if not args.zip.is_file():
        raise SystemExit(f"no archive at {args.zip}")

    with zipfile.ZipFile(args.zip) as archive:
        cases = index_cases(archive)

        if args.list:
            show(cases)
            return

        chosen = sorted(cases)[: args.cases]
        args.out.mkdir(parents=True, exist_ok=True)
        written = extract(archive, cases, chosen, args.structures, args.out)

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps(
            {
                "source": "Totalsegmentator_dataset_v201 (Zenodo 10047292, CC BY 4.0)",
                "cases": chosen,
                "structures": args.structures,
            },
            indent=2,
        )
        + "\n"
    )

    print(f"extracted {len(chosen)} cases, {human(written)}")
    print(f"manifest  {MANIFEST}")
    print()
    print("The zip can go once this looks right — the manifest rebuilds the")
    print("same subset from a fresh download.")


if __name__ == "__main__":
    main()
