from pathlib import Path


# ---------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "dataset" / "pilot"

TRAIN_IMAGES = DATASET_DIR / "train" / "images"
TRAIN_LABELS = DATASET_DIR / "train" / "labels"

VALID_IMAGES = DATASET_DIR / "valid" / "images"
VALID_LABELS = DATASET_DIR / "valid" / "labels"

TEST_IMAGES = DATASET_DIR / "test" / "images"
TEST_LABELS = DATASET_DIR / "test" / "labels"

CLASS_COUNT = 10


# ---------------------------------------------------------
# IMAGE EXTENSIONS
# ---------------------------------------------------------

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def get_images(directory):
    if not directory.exists():
        return []

    return [
        p for p in directory.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def get_labels(directory):
    if not directory.exists():
        return []

    return [
        p for p in directory.iterdir()
        if p.is_file()
        and p.suffix.lower() == ".txt"
    ]


def validate_split(name, image_dir, label_dir):

    print("\n" + "-" * 70)
    print(f"{name.upper()} DATASET")
    print("-" * 70)

    images = get_images(image_dir)
    labels = get_labels(label_dir)

    print(f"Images: {len(images)}")
    print(f"Labels: {len(labels)}")

    errors = 0

    # -----------------------------------------------------
    # IMAGE -> LABEL MATCHING
    # -----------------------------------------------------

    image_stems = {
        image.stem
        for image in images
    }

    label_stems = {
        label.stem
        for label in labels
    }

    missing_labels = image_stems - label_stems
    orphan_labels = label_stems - image_stems

    if missing_labels:

        errors += len(missing_labels)

        print("\n❌ Missing labels:")

        for name in sorted(missing_labels):
            print(f"   {name}")

    if orphan_labels:

        errors += len(orphan_labels)

        print("\n❌ Labels without images:")

        for name in sorted(orphan_labels):
            print(f"   {name}")

    if not missing_labels and not orphan_labels:

        print("✅ Image/label matching passed")

    # -----------------------------------------------------
    # LABEL CONTENT VALIDATION
    # -----------------------------------------------------

    class_counts = {}

    for label_file in labels:

        try:

            content = label_file.read_text(
                encoding="utf-8"
            ).strip()

        except Exception as exc:

            errors += 1

            print(
                f"\n❌ Could not read: "
                f"{label_file.name}"
            )

            print(f"   {exc}")

            continue

        # Empty labels are allowed technically,
        # but report them clearly.

        if not content:

            print(
                f"⚠️ Empty label file: "
                f"{label_file.name}"
            )

            continue

        lines = content.splitlines()

        for line_number, line in enumerate(
            lines,
            start=1
        ):

            parts = line.split()

            # YOLO format:
            # class x_center y_center width height

            if len(parts) != 5:

                errors += 1

                print(
                    f"\n❌ Invalid format:"
                    f" {label_file.name}"
                    f" line {line_number}"
                )

                print(f"   {line}")

                continue

            try:

                class_id = int(parts[0])

                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

            except ValueError:

                errors += 1

                print(
                    f"\n❌ Non-numeric label:"
                    f" {label_file.name}"
                    f" line {line_number}"
                )

                print(f"   {line}")

                continue

            # -------------------------------------------------
            # CLASS ID
            # -------------------------------------------------

            if not (
                0 <= class_id < CLASS_COUNT
            ):

                errors += 1

                print(
                    f"\n❌ Invalid class ID:"
                    f" {label_file.name}"
                    f" line {line_number}"
                )

                print(
                    f"   Class ID: {class_id}"
                )

            else:

                class_counts[class_id] = (
                    class_counts.get(
                        class_id,
                        0
                    )
                    + 1
                )

            # -------------------------------------------------
            # COORDINATES
            # -------------------------------------------------

            coordinates = [
                x_center,
                y_center,
                width,
                height,
            ]

            if not all(
                0.0 <= value <= 1.0
                for value in coordinates
            ):

                errors += 1

                print(
                    f"\n❌ Coordinate out of range:"
                    f" {label_file.name}"
                    f" line {line_number}"
                )

                print(
                    f"   "
                    f"x={x_center}, "
                    f"y={y_center}, "
                    f"w={width}, "
                    f"h={height}"
                )

            # -------------------------------------------------
            # WIDTH / HEIGHT
            # -------------------------------------------------

            if width <= 0 or height <= 0:

                errors += 1

                print(
                    f"\n❌ Invalid box size:"
                    f" {label_file.name}"
                    f" line {line_number}"
                )

    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    print("\nClass IDs detected:")

    for class_id in sorted(class_counts):

        print(
            f"  {class_id}: "
            f"{class_counts[class_id]} object(s)"
        )

    if errors == 0:

        print(
            "\n✅ "
            f"{name} validation PASSED"
        )

    else:

        print(
            f"\n❌ {name} validation FAILED"
        )

    return errors


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("PHASE 2C - PILOT YOLO DATASET VALIDATION")
    print("=" * 70)

    print(
        f"\nDataset:"
        f"\n{DATASET_DIR}"
    )

    total_errors = 0

    total_errors += validate_split(
        "Train",
        TRAIN_IMAGES,
        TRAIN_LABELS,
    )

    total_errors += validate_split(
        "Validation",
        VALID_IMAGES,
        VALID_LABELS,
    )

    total_errors += validate_split(
        "Test",
        TEST_IMAGES,
        TEST_LABELS,
    )

    # -----------------------------------------------------
    # FINAL RESULT
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL VALIDATION")
    print("=" * 70)

    if total_errors == 0:

        print(
            "\n✅ PILOT YOLO DATASET VALIDATION PASSED"
        )

        print(
            "All images and YOLO labels are valid."
        )

        print(
            "\nReady for pilot training."
        )

    else:

        print(
            f"\n❌ PILOT YOLO DATASET VALIDATION FAILED"
        )

        print(
            f"Total errors: {total_errors}"
        )

        print(
            "\nFix the errors before training."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()