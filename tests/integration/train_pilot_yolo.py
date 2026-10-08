from pathlib import Path
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_YAML = PROJECT_ROOT / "dataset" / "pilot" / "data.yaml"

MODEL = "yolo11n.pt"

print("=" * 70)
print("PHASE 2D - PILOT YOLO TRAINING")
print("=" * 70)

print(f"\nDataset:")
print(DATA_YAML)

print(f"\nBase model:")
print(MODEL)

print("\nLoading YOLO model...")

model = YOLO(MODEL)

print("YOLO model loaded successfully.")

print("\nStarting pilot training...")
print("-" * 70)

results = model.train(
    data=str(DATA_YAML),

    epochs=50,

    imgsz=640,

    batch=4,

    workers=0,

    project=str(PROJECT_ROOT / "runs" / "detect"),

    name="pilot_indian_traffic",

    pretrained=True,

    patience=15,

    save=True,

    plots=True,

    verbose=True,
)

print("\n" + "=" * 70)
print("PILOT TRAINING COMPLETED")
print("=" * 70)

print("\nTraining output:")
print(
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "pilot_indian_traffic"
)

print("\nBest model should be:")
print(
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "pilot_indian_traffic"
    / "weights"
    / "best.pt"
)

print("=" * 70)