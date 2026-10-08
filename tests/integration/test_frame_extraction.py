from pathlib import Path
import sys


# ---------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.perception.video.frame_extractor import (
    VideoFrameExtractor,
)


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "video"
    / "traffic_test.mp4"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "raw"
    / "cctv"
    / "traffic_test"
)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("PHASE 2B - CCTV FRAME EXTRACTION")
    print("=" * 70)

    print(
        f"\nInput video:\n{VIDEO_PATH}"
    )

    print(
        f"\nOutput directory:\n{OUTPUT_DIR}"
    )

    # -----------------------------------------------------
    # EXTRACT
    # -----------------------------------------------------

    extractor = VideoFrameExtractor(
        video_path=str(VIDEO_PATH),
        output_dir=str(OUTPUT_DIR),
        interval_seconds=1.0,
        image_format="jpg",
    )

    count = extractor.extract()

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("FRAME EXTRACTION SUMMARY")
    print("=" * 70)

    print(
        f"Images extracted: {count}"
    )

    print(
        f"Saved to: {OUTPUT_DIR}"
    )

    print("\n" + "=" * 70)
    print(
        "FRAME EXTRACTION COMPLETED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()