import json
import joblib
from pathlib import Path

ARTIFACTS_DIR = Path("artifacts")

for model_dir in ARTIFACTS_DIR.iterdir():
    if not model_dir.is_dir():
        continue

    model_path = model_dir / "model.joblib"
    if not model_path.exists():
        print(f"[SKIP] {model_dir.name} — model.joblib not found")
        continue

    serving_model_path = model_dir / "serving_model.joblib"
    metadata_path = model_dir / "metadata.json"

    if serving_model_path.exists() and metadata_path.exists():
        print(f"[SKIP] {model_dir.name} — serving artifacts already exist")
        continue

    print(f"[Processing] {model_dir.name}...")

    wrapper = joblib.load(model_path)

    joblib.dump(wrapper.model, serving_model_path)
    print(f"  saved: {serving_model_path}")

    model_type = model_dir.name.replace("_model", "")
    metadata = {
        "model_type": model_type,
        "feature_names": wrapper.features_,
        "metrics": {},
        "thresholds": {},
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  saved: {metadata_path}")

print("\nDone. Copy serving_model.joblib + metadata.json to the web service.")
