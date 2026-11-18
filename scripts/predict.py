"""Run the trained model on prepared scans and save what it outputs.

Run:  python scripts/predict.py --data PREPARED_FOLDER --model best.pt --out predictions/test --group test

For each scan this gives the model the fake X-ray of that scan and saves the model's output in one
file, SCAN_NUMBER.npz, with:
  hu      the brightness of every point of the block, in Hounsfield units, [rows, columns, depth]
  labels  the part number of every point of the block (0 background, 1 lower jaw, 2 upper jaw,
          3 nerve canal, 4 sinus, 5 teeth), [rows, columns, depth]
This is the same layout as the prepared scan files, so the two can be compared point by point.
The comparing is done by scripts/measure.py.

--group says which scans to use:
  test   the 10 scans that were never used during training (the default)
  check  the 5 scans that were used during training to pick the best saved model
  train  the 46 scans the model was trained on
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model import XrayTo3D
from src.train_data import make_split, unit_to_hu


def load_model(path, device):
    """Load a model file saved by scripts/train.py."""
    saved = torch.load(path, map_location=device, weights_only=False)  # our own file, so it is safe to load fully
    widths = (16, 32, 48, 64, 64) if saved.get("small") else (48, 96, 192, 384, 512)
    model = XrayTo3D(depth=saved["depth"], widths=widths).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    return model, saved


@torch.no_grad()
def predict_one(model, xray, device):
    """xray: [rows, columns] numbers from 0 to 1. Returns (hu, labels), both [rows, columns, depth]."""
    x = torch.from_numpy(xray.astype(np.float32))[None, None].to(device)
    with torch.autocast(device.type, enabled=device.type == "cuda"):
        bright, scores = model(x)
    hu = unit_to_hu(bright[0].float()).round().clamp(-1000, 3000).to(torch.int16)
    labels = scores[0].argmax(dim=0).to(torch.uint8)
    return hu.permute(1, 2, 0).cpu().numpy(), labels.permute(1, 2, 0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder made by prepare_tf3.py")
    ap.add_argument("--model", required=True, help="model file saved by train.py, for example best.pt")
    ap.add_argument("--out", required=True, help="folder for the model's outputs")
    ap.add_argument("--group", default="test", choices=["test", "check", "train"])
    ap.add_argument("--scans", nargs="*", help="scan numbers to use instead of a whole group")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"running on: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""), flush=True)

    numbers = args.scans if args.scans else make_split(args.data)[args.group]

    # The training script saved the list of scans it trained on next to the model file.
    # Stop if a scan we are about to score as "never used" was in fact used.
    settings_path = Path(args.model).parent / "settings.json"
    if settings_path.exists():
        settings = json.loads(settings_path.read_text())
        used = set(settings["train"]) | (set(settings["check"]) if args.group == "test" else set())
        if args.group != "train":
            overlap = sorted(set(numbers) & used)
            if overlap:
                raise SystemExit(f"ERROR: these scans were used during training: {overlap}")
            print(f"checked against {settings_path.name}: none of the {len(numbers)} scans was used for training", flush=True)
    else:
        print("no settings.json next to the model file, so the scan list could not be checked", flush=True)

    model, saved = load_model(args.model, device)
    print(f"loaded {args.model} (saved after epoch {saved['epoch']})", flush=True)
    for i, num in enumerate(numbers):
        t0 = time.time()
        with np.load(Path(args.data) / f"{num}.npz") as f:
            xray = f["xray"]
        hu, labels = predict_one(model, xray, device)
        np.savez_compressed(out / f"{num}.npz", hu=hu, labels=labels)
        print(f"[{i + 1}/{len(numbers)}] scan {num} done in {time.time() - t0:.1f}s", flush=True)
    (out / "about.json").write_text(json.dumps(dict(model=str(args.model), epoch=saved["epoch"], group="given list" if args.scans else args.group, scans=list(numbers)), indent=1))
    print(f"{len(numbers)} outputs are in {out}")


if __name__ == "__main__":
    main()
