from pathlib import Path
import shutil
import sys


# ---------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------
# INPUT DIRECTORIES
# ---------------------------------------------------------

CCTV_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "raw"
    / "cctv"
)

EXTERNAL_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "raw"
    / "external"
    / "standardized"
)


# ---------------------------------------------------------
# OUTPUT DIRECTORIES
# ---------------------------------------------------------

PREPARED_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "prepared"
)

PREPARED_CCTV_DIR = (
    PREPARED_DIR
    / "images"
    / "cctv"
)

PREPARED_EXTERNAL_DIR = (
    PREPARED_DIR
    / "images"
    / "external"
)


# ---------------------------------------------------------
# DUPLICATES TO EXCLUDE
# ---------------------------------------------------------

DUPLICATE_FILES = {
    "external_000040.jpg",
    "external_000159.jpg",
    "external_000162.jpg",
    "external_000193.jpg",
    "external_000224.jpg",
    "external_000286.jpg",
}


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def get_images(directory: Path):

    if not directory.exists():
        return []

    supported = {
        ".jpg",
        ".jpeg",
        ".png",
    }

    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in supported
    )


def copy_images(
    source_dir: Path,
    output_dir: Path,
    exclude_names=None,
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    images = get_images(source_dir)

    copied = 0
    excluded = 0

    for source in images:

        if (
            exclude_names
            and source.name in exclude_names
        ):
            excluded += 1
            continue

        destination = (
            output_dir
            / source.name
        )

        shutil.copy2(
            source,
            destination,
        )

        copied += 1

    return copied, excluded


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("PHASE 2B - DATASET PREPARATION")
    print("=" * 70)

    # -----------------------------------------------------
    # CREATE OUTPUT DIRECTORIES
    # -----------------------------------------------------

    PREPARED_CCTV_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PREPARED_EXTERNAL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # CCTV
    # -----------------------------------------------------

    cctv_copied, cctv_excluded = copy_images(
        CCTV_DIR,
        PREPARED_CCTV_DIR,
    )

    # -----------------------------------------------------
    # EXTERNAL
    # -----------------------------------------------------

    external_copied, external_excluded = copy_images(
        EXTERNAL_DIR,
        PREPARED_EXTERNAL_DIR,
        exclude_names=DUPLICATE_FILES,
    )

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    total_copied = (
        cctv_copied
        + external_copied
    )

    total_excluded = (
        cctv_excluded
        + external_excluded
    )

    print("\n" + "=" * 70)
    print("PREPARATION SUMMARY")
    print("=" * 70)

    print(
        f"\nCCTV images copied:      "
        f"{cctv_copied}"
    )

    print(
        f"External images copied:  "
        f"{external_copied}"
    )

    print(
        f"Total prepared:          "
        f"{total_copied}"
    )

    print(
        f"Duplicates excluded:     "
        f"{total_excluded}"
    )

    print(
        f"\nPrepared dataset:"
        f"\n{PREPARED_DIR}"
    )

    print("\n" + "=" * 70)
    print(
        "DATASET PREPARATION COMPLETED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()