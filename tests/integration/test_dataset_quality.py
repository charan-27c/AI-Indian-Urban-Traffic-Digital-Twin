from pathlib import Path
import sys
import hashlib

import cv2


# ---------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------
# DATASET DIRECTORIES
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
# SETTINGS
# ---------------------------------------------------------

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


MIN_WIDTH = 320
MIN_HEIGHT = 240


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def get_images(directory: Path):

    if not directory.exists():
        return []

    images = []

    for path in directory.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            images.append(path)

    return sorted(images)


def calculate_hash(path: Path):

    hasher = hashlib.md5()

    with path.open("rb") as file:

        while True:

            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            hasher.update(chunk)

    return hasher.hexdigest()


def inspect_images(images):

    readable = []
    invalid = []
    too_small = []

    hashes = {}

    duplicates = []

    for image_path in images:

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            invalid.append(image_path)

            continue

        height, width = image.shape[:2]

        if (
            width < MIN_WIDTH
            or height < MIN_HEIGHT
        ):

            too_small.append(
                (
                    image_path,
                    width,
                    height,
                )
            )

        readable.append(
            (
                image_path,
                width,
                height,
            )
        )

        image_hash = calculate_hash(
            image_path
        )

        if image_hash in hashes:

            duplicates.append(
                (
                    image_path,
                    hashes[image_hash],
                )
            )

        else:

            hashes[image_hash] = image_path

    return (
        readable,
        invalid,
        too_small,
        duplicates,
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("PHASE 2B - DATASET QUALITY INSPECTION")
    print("=" * 70)

    # -----------------------------------------------------
    # DISCOVER
    # -----------------------------------------------------

    cctv_images = get_images(
        CCTV_DIR
    )

    external_images = get_images(
        EXTERNAL_DIR
    )

    all_images = (
        cctv_images
        + external_images
    )

    print("\nDataset sources:")
    print(
        f"  CCTV images:     "
        f"{len(cctv_images)}"
    )
    print(
        f"  External images: "
        f"{len(external_images)}"
    )
    print(
        f"  Total images:    "
        f"{len(all_images)}"
    )

    # -----------------------------------------------------
    # INSPECT
    # -----------------------------------------------------

    (
        readable,
        invalid,
        too_small,
        duplicates,
    ) = inspect_images(
        all_images
    )

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("DATASET QUALITY SUMMARY")
    print("=" * 70)

    print(
        f"\nTotal images:       "
        f"{len(all_images)}"
    )

    print(
        f"Readable:           "
        f"{len(readable)}"
    )

    print(
        f"Invalid:             "
        f"{len(invalid)}"
    )

    print(
        f"Too small:           "
        f"{len(too_small)}"
    )

    print(
        f"Duplicate images:    "
        f"{len(duplicates)}"
    )

    usable_count = (
        len(readable)
        - len(too_small)
        - len(duplicates)
    )

    print(
        f"Potentially usable: "
        f"{max(usable_count, 0)}"
    )

    # -----------------------------------------------------
    # INVALID
    # -----------------------------------------------------

    if invalid:

        print(
            "\n" + "-" * 70
        )

        print("INVALID IMAGES")

        for path in invalid:

            print(
                f"  {path}"
            )

    # -----------------------------------------------------
    # SMALL
    # -----------------------------------------------------

    if too_small:

        print(
            "\n" + "-" * 70
        )

        print("SMALL IMAGES")

        for (
            path,
            width,
            height,
        ) in too_small:

            print(
                f"  {path.name} "
                f"({width}x{height})"
            )

    # -----------------------------------------------------
    # DUPLICATES
    # -----------------------------------------------------

    if duplicates:

        print(
            "\n" + "-" * 70
        )

        print("DUPLICATE IMAGES")

        for (
            duplicate,
            original,
        ) in duplicates:

            print(
                f"  {duplicate.name}"
            )

            print(
                f"    same as: "
                f"{original.name}"
            )

    # -----------------------------------------------------
    # SAMPLE DIMENSIONS
    # -----------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print("SAMPLE IMAGE DIMENSIONS")

    for (
        path,
        width,
        height,
    ) in readable[:10]:

        print(
            f"  {path.name}: "
            f"{width} x {height}"
        )

    # -----------------------------------------------------
    # COMPLETION
    # -----------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "DATASET QUALITY INSPECTION COMPLETED"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()