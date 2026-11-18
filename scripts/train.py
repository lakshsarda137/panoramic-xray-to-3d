"""Train the model that turns a panoramic X-ray into a straightened 3D block.

Run:  python scripts/train.py --data data/prepared --out runs/first --epochs 150

Words used here:
  epoch  = one pass in which the model is shown as many training pieces as there are training scans, times 4.
  step   = the model looks at a few pieces, its output is compared with the correct answer, and its
           numbers are changed a little to make the output closer to the answer.
  loss   = one number that says how far the output is from the correct answer. Lower is better.
  Dice   = overlap between a predicted part and the true part. 0 = no overlap, 1 = the same.

Pause and continue:
  After every epoch the script saves everything it needs in the file last.pt in the --out folder.
  To pause, stop the script (in Colab: the stop button of the cell). To continue, run the same
  command again. It starts after the last saved epoch. At most one epoch of work (about 80 seconds
  on Colab) is lost. The same happens if Colab stops by itself.
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.curved_volume import DEPTH_IN_MM, DEPTH_STEP_MM, FRAME_BITE_ROW, VOXEL_MM
from src.model import XrayTo3D
from src.train_data import OUTSIDE, Examples, change_xray_look, hu_to_unit, make_split, unit_to_hu

CLASS_NAMES = ["background", "lower jaw", "upper jaw", "nerve canal", "sinus", "teeth"]
LABEL_COLORS = np.array([[0, 0, 0], [70, 110, 200], [60, 170, 170], [255, 40, 40], [255, 220, 0], [240, 240, 240]], np.uint8)
# The nerve canal is very small, so mistakes on it count more.
CLASS_WEIGHTS = [1.0, 1.0, 1.0, 4.0, 2.0, 1.0]


def loss_fn(bright, scores, hu, labels):
    """bright [b, d, r, c] 0..1; scores [b, classes, d, r, c]; hu and labels [b, d, r, c]."""
    inside = labels != OUTSIDE
    n_inside = inside.sum().clamp(min=1)
    l_bright = ((bright - hu_to_unit(hu.float())).abs() * inside).sum() / n_inside
    target = labels.long()
    weights = torch.tensor(CLASS_WEIGHTS, device=scores.device, dtype=scores.dtype)
    l_part = F.cross_entropy(scores, target, weight=weights, ignore_index=OUTSIDE)
    # Dice part of the loss: overlap of each predicted part with the true part
    prob = scores.softmax(dim=1)
    misses = []
    for c in range(1, scores.shape[1]):
        true_c = (target == c) & inside
        if true_c.any():
            p = prob[:, c] * inside
            misses.append(1 - (2 * (p * true_c).sum() + 1) / (p.sum() + true_c.sum() + 1))
    l_overlap = torch.stack(misses).mean() if misses else scores.sum() * 0
    return 10 * l_bright + l_part + l_overlap, dict(bright=l_bright.item(), part=l_part.item(), overlap=l_overlap.item())


@torch.no_grad()
def check(model, examples, device, use_half):
    """Run the model on whole scans it was not trained on. Returns average numbers and the last output."""
    model.eval()
    err_hu, psnr, dice = [], [], {c: [] for c in range(1, 6)}
    for i in range(len(examples)):
        xray, hu, labels = (t.to(device) for t in examples.whole(i))
        with torch.autocast(device.type, enabled=use_half):
            bright, scores = model(xray)
        bright, pred = bright.float(), scores.argmax(dim=1)
        inside = labels != OUTSIDE
        diff = (bright - hu_to_unit(hu.float()))[inside]
        err_hu.append(diff.abs().mean().item() * 4000)
        psnr.append(-10 * math.log10(diff.pow(2).mean().item()))
        for c in range(1, 6):
            t, p = (labels == c) & inside, (pred == c) & inside
            if t.any():
                dice[c].append((2 * (t & p).sum() / (t.sum() + p.sum())).item())
    model.train()
    out = dict(err_hu=float(np.mean(err_hu)), psnr=float(np.mean(psnr)))
    for c in range(1, 6):
        out[f"dice {CLASS_NAMES[c]}"] = float(np.mean(dice[c])) if dice[c] else float("nan")
    return out, (xray[0, 0].cpu().numpy(), unit_to_hu(bright[0]).cpu().numpy(), pred[0].cpu().numpy(),
                 hu[0].cpu().numpy(), labels[0].cpu().numpy())


def save_picture(path, title, xray, pred_hu, pred_lab, true_hu, true_lab):
    """Predicted block next to the true block: one cut along the jaw curve and one cut seen from above."""
    outside = true_lab == OUTSIDE  # places outside the scan have no correct answer, so blank them in the output too
    pred_hu, pred_lab = np.where(outside, -1000, pred_hu), np.where(outside, OUTSIDE, pred_lab)
    at_curve = int(round(DEPTH_IN_MM / DEPTH_STEP_MM))  # the depth sample that is on the jaw curve
    row = FRAME_BITE_ROW + int(12 / VOXEL_MM)            # 12 mm below where the teeth meet
    color = lambda lab: LABEL_COLORS[np.minimum(lab, 5)] * (lab[..., None] != OUTSIDE)
    fig, ax = plt.subplots(5, 2, figsize=(18, 17), gridspec_kw=dict(height_ratios=[3, 3, 3, 1.2, 1.2]))
    ax[0, 0].imshow(xray, cmap="gray", vmin=0, vmax=1); ax[0, 0].set_title("input: X-ray")
    ax[0, 1].axis("off"); ax[0, 1].text(0.02, 0.5, title, fontsize=13, va="center")
    for col, (hu, lab, name) in enumerate([(pred_hu, pred_lab, "MODEL OUTPUT"), (true_hu, true_lab, "CORRECT ANSWER")]):
        ax[1, col].imshow(hu[at_curve], cmap="gray", vmin=-1000, vmax=2500); ax[1, col].set_title(f"{name}: cut along the jaw curve")
        ax[2, col].imshow(color(lab[at_curve])); ax[2, col].set_title(f"{name}: parts (blue lower jaw, teal upper jaw, red nerve canal, yellow sinus, white teeth)")
        ax[3, col].imshow(hu[:, row], cmap="gray", vmin=-1000, vmax=2500, aspect=1.25); ax[3, col].set_title(f"{name}: seen from above, 12 mm below the bite")
        ax[4, col].imshow(color(lab[:, row]), aspect=1.25)
    for a in ax.ravel():
        a.axis("off")
    plt.tight_layout(); plt.savefig(path, dpi=55); plt.close(fig)


PROGRESS = dict(saved_epochs=0, total=0)  # used for the message that is printed when the script is paused


def save_safely(state, path):
    """Write to a second file first, then rename it. If the script is stopped in the middle of
    writing, the old file is still whole."""
    tmp = Path(str(path) + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder made by prepare_tf3.py")
    ap.add_argument("--out", required=True, help="folder for saved models, the log and pictures")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=2, help="pieces per step")
    ap.add_argument("--piece-cols", type=int, default=384, help="width of one training piece")
    ap.add_argument("--lr", type=float, default=5e-4, help="how big each change to the model is")
    ap.add_argument("--limit", type=int, default=0, help="use only this many training scans (for a quick test)")
    ap.add_argument("--small", action="store_true", help="use a much smaller model (only for a quick test on a laptop)")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_half = device.type == "cuda"  # half-size numbers on the graphics card: faster and uses less memory
    print(f"running on: {device}" + (f" ({torch.cuda.get_device_name(0)})" if use_half else ""), flush=True)

    split = make_split(args.data)
    train_nums = split["train"][:args.limit] if args.limit else split["train"]
    check_nums = split["check"][:max(1, args.limit)] if args.limit else split["check"]
    t0 = time.time()
    train, checks = Examples(args.data, train_nums), Examples(args.data, check_nums)
    print(f"loaded {len(train)} training scans and {len(checks)} check scans in {time.time() - t0:.0f}s", flush=True)

    depth = train.hu.shape[1]
    model = XrayTo3D(depth=depth, widths=(16, 32, 48, 64, 64) if args.small else (48, 96, 192, 384, 512)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(enabled=use_half)
    steps_per_epoch = max(1, len(train) * 4 // args.batch)
    start, best = 0, -1.0
    last_path = out / "last.pt"
    if last_path.exists():
        saved = torch.load(last_path, map_location=device, weights_only=False)  # our own file, so it is safe to load fully
        model.load_state_dict(saved["model"]); opt.load_state_dict(saved["opt"])
        if "scaler" in saved:
            scaler.load_state_dict(saved["scaler"])
        start, best = saved["epoch"], saved["best"]
        print(f"found saved work: {start} of {args.epochs} epochs are finished. Continuing with epoch {start + 1}.", flush=True)
    else:
        print("no saved work found in this folder. Starting at epoch 1.", flush=True)
    PROGRESS.update(saved_epochs=start, total=args.epochs)
    (out / "settings.json").write_text(json.dumps(dict(vars(args), train=train_nums, check=check_nums), indent=1))

    for epoch in range(start, args.epochs):
        lr = args.lr * 0.5 * (1 + math.cos(math.pi * epoch / args.epochs))  # start big, end small
        for g in opt.param_groups:
            g["lr"] = max(lr, 1e-5)
        t0, sums = time.time(), {}
        for _ in range(steps_per_epoch):
            xray, hu, labels = (t.to(device) for t in train.random_batch(args.batch, args.piece_cols))
            xray = change_xray_look(xray)
            with torch.autocast(device.type, enabled=use_half):
                bright, scores = model(xray)
            loss, parts = loss_fn(bright.float(), scores.float(), hu, labels)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            parts["loss"] = loss.item()
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v
        row = dict(epoch=epoch + 1, seconds=round(time.time() - t0), **{f"train {k}": v / steps_per_epoch for k, v in sums.items()})
        res, shown = check(model, checks, device, use_half)
        row.update(res)
        score = float(np.nanmean([res[f"dice {CLASS_NAMES[c]}"] for c in range(1, 6)]))
        if use_half:
            row["gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 1)
        print(f"epoch {epoch + 1}/{args.epochs} ({row['seconds']}s" + (f", {row['gpu_gb']} GB of graphics card memory" if use_half else "") + f") loss {row['train loss']:.3f} | check scans: "
              f"error {res['err_hu']:.0f} HU, PSNR {res['psnr']:.1f}, Dice " +
              ", ".join(f"{CLASS_NAMES[c]} {res[f'dice {CLASS_NAMES[c]}']:.2f}" for c in range(1, 6)), flush=True)
        new_file = not (out / "log.csv").exists()
        with open(out / "log.csv", "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new_file:
                w.writeheader()
            w.writerow(row)
        state = dict(model=model.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(), epoch=epoch + 1,
                     best=max(best, score), depth=depth, small=args.small)
        if score > best:
            best = score
            save_safely(state, out / "best.pt")
        save_safely(state, last_path)
        PROGRESS["saved_epochs"] = epoch + 1
        if (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs or epoch == 0:
            save_picture(out / f"check_epoch_{epoch + 1:03d}.png", f"scan {checks.numbers[-1]} (not used for training)\nafter epoch {epoch + 1}", *shown)
    print(f"finished. Best average Dice on check scans: {best:.3f}. Files are in {out}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:  # the stop button
        print(f"\nPAUSED. {PROGRESS['saved_epochs']} of {PROGRESS['total']} epochs are finished and saved. "
              "To continue, run this same cell (or command) again.", flush=True)
