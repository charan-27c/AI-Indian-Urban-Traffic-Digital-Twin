from pathlib import Path
import sys


# ---------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.perception.dataset.image_standardizer import (
    ImageStandardizer,
)


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

INPUT_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "raw"
    / "external"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "raw"
    / "external"
    / "standardized"
)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("PHASE 2B - EXTERNAL IMAGE STANDARDIZATION")
    print("=" * 70)

    print(
        f"\nInput:\n{INPUT_DIR}"
    )

    print(
        f"\nOutput:\n{OUTPUT_DIR}"
    )

    standardizer = ImageStandardizer(
        input_dir=str(INPUT_DIR),
        output_dir=str(OUTPUT_DIR),
        jpeg_quality=95,
    )

    count = standardizer.standardize()

    print("\n" + "=" * 70)
    print("STANDARDIZATION SUMMARY")
    print("=" * 70)

    print(
        f"Images converted: {count}"
    )

    print(
        f"Output directory: {OUTPUT_DIR}"
    )

    print("\n" + "=" * 70)
    print(
        "IMAGE STANDARDIZATION COMPLETED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()