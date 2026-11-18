"""Build three notebooks that hold copies of the project code files:
  notebooks/xray_to_3d_colab.ipynb                  trains, for Google Colab (also prepares the scans)
  notebooks/kaggle/xray_to_3d_kaggle.ipynb          trains, for Kaggle (reads scans that were prepared on the Mac)
  notebooks/kaggle_scoring/xray_to_3d_scoring.ipynb does not train; runs the finished model on the test
                                                    scans and measures the result in millimeters (Kaggle)

Run again after any change to the copied files:
    .venv/bin/python scripts/build_colab_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "notebooks" / "xray_to_3d_colab.ipynb"

EMBEDDED_FILES = [
    "src/fake_panoramic.py",
    "src/tf3_io.py",
    "src/curved_volume.py",
    "src/model.py",
    "src/train_data.py",
    "scripts/prepare_tf3.py",
    "scripts/train.py",
]


def to_lines(text):
    """Split text into the notebook 'source' format: every line except the
    last ends with a newline character."""
    return text.splitlines(keepends=True)


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": to_lines(text.strip("\n"))}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": to_lines(text),
    }


def writefile_cell(rel_path, folder="/content/project"):
    text = (ROOT / rel_path).read_text(encoding="utf-8")
    for number, line in enumerate(text.splitlines(), start=1):
        if line.startswith("%%"):
            raise SystemExit(
                f"ERROR: {rel_path} line {number} starts with '%%'. "
                "That would break the %%writefile cell. Change that line and build again."
            )
    return code(f"%%writefile {folder}/{rel_path}\n" + text)


INTRO = """
# Panoramic X-ray to 3D scan: training

This notebook copies the training scans from Google Drive.
It makes a fake panoramic X-ray and a straightened 3D block from each scan.
It trains a model that gets the X-ray as input and outputs the 3D block.
It saves the model and pictures to Google Drive in the folder `xray_construction/runs/`.
Before running, choose Runtime > Change runtime type > T4 GPU.
Then run the cells from top to bottom.

**To pause the training:** click the stop button (the square) at the left of the training cell.
The notebook prints `PAUSED` and how many epochs are saved. You can then close the tab.

**To continue:** open this notebook again and click "Run all". Do not change `RUN_NAME`.
The training continues after the last saved epoch. At most one epoch (about 80 seconds) is lost.
The same works if Colab stops by itself.
"""

SETTINGS = """EPOCHS = 2            # how many passes over the training scans. 2 is a short test. Use 150 for the full run.
TRAIN_SCANS = 0       # 0 = use all training scans. A small number, for example 6, makes a faster test.
RUN_NAME = "test"     # name of the folder on Google Drive where results are saved
ZIP_NAME = "tf3F_0.4mm.zip"   # the file with the training scans on Google Drive"""

GPU_CHECK = """import torch
assert torch.cuda.is_available(), "No graphics card. Choose Runtime > Change runtime type > T4 GPU, then run again."
print(torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")"""

DRIVE = """import glob, os, subprocess
from google.colab import drive
drive.mount("/content/drive")
# After the scans are prepared once, a copy is kept on Google Drive. Later runs use that copy and skip the 10 minutes of preparing.
SAVED_PREPARED = "/content/drive/MyDrive/xray_construction/prepared_scans.zip"
if os.path.exists(SAVED_PREPARED) or glob.glob("/content/prepared/*.npz"):
    DATA = None
    print("The prepared scans already exist. The original scans are not needed.")
else:
    hits = glob.glob("/content/drive/MyDrive/" + ZIP_NAME) or glob.glob("/content/drive/MyDrive/**/" + ZIP_NAME, recursive=True)
    assert hits, ZIP_NAME + " was not found in this Google Drive. Check that this is the Google account that holds the file."
    if not glob.glob("/content/data/**/imagesTr", recursive=True):
        subprocess.run(["unzip", "-q", "-o", hits[0], "-d", "/content/data"], check=True)
    DATA = os.path.dirname(glob.glob("/content/data/**/imagesTr", recursive=True)[0])
    print("zip file:", hits[0])
    print("scans:", len(os.listdir(DATA + "/imagesTr")), " label files:", len(os.listdir(DATA + "/labelsTr")))"""

FOLDERS = """import os
os.makedirs("/content/project/src", exist_ok=True)
os.makedirs("/content/project/scripts", exist_ok=True)
!pip -q install nibabel"""

PREPARE = """import shutil
if glob.glob("/content/prepared/*.npz"):
    print("The prepared scans are already in Colab.")
elif os.path.exists(SAVED_PREPARED):
    shutil.unpack_archive(SAVED_PREPARED, "/content/prepared")
    print("Copied the prepared scans from Google Drive:", len(glob.glob("/content/prepared/*.npz")), "scans")
else:
    !python /content/project/scripts/prepare_tf3.py --data "$DATA" --out /content/prepared
if not os.path.exists(SAVED_PREPARED):
    shutil.make_archive("/content/prepared_scans", "zip", "/content/prepared")
    os.makedirs(os.path.dirname(SAVED_PREPARED), exist_ok=True)
    shutil.copy("/content/prepared_scans.zip", SAVED_PREPARED + ".tmp")
    os.replace(SAVED_PREPARED + ".tmp", SAVED_PREPARED)
    os.remove("/content/prepared_scans.zip")
    print("Saved a copy of the prepared scans to Google Drive:", round(os.path.getsize(SAVED_PREPARED) / 1e6), "MB")"""

PREVIEW = """from IPython.display import Image, display
display(Image("/content/prepared/previews/sheet_01.png", width=900))"""

TRAIN = """OUT = "/content/drive/MyDrive/xray_construction/runs/" + RUN_NAME
!python /content/project/scripts/train.py --data /content/prepared --out "$OUT" --epochs $EPOCHS --limit $TRAIN_SCANS"""

RESULTS = """import glob, pandas as pd
from IPython.display import Image, display
display(pd.read_csv(OUT + "/log.csv").round(3).tail(10))
display(Image(sorted(glob.glob(OUT + "/check_epoch_*.png"))[-1], width=1100))"""


METADATA = {
    "accelerator": "GPU",
    "colab": {"provenance": [], "gpuType": "T4"},
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "language_info": {"name": "python"},
}

# ---------------------------------------------------------------- Kaggle
KAGGLE_USER = "lakshsarda"
KAGGLE_NOTEBOOK = "xray-to-3d-training"
KAGGLE_DATASET = "xray-prepared-scans"
KAGGLE_DIR = ROOT / "notebooks" / "kaggle"

K_INTRO = f"""
# Panoramic X-ray to 3D scan: training (Kaggle)

This notebook trains a model that gets a panoramic X-ray as input and outputs a straightened 3D block of the jaw.
It reads the prepared scans from the private Kaggle dataset `{KAGGLE_DATASET}`.

**To start the full training:** the first code cell is already set for it (`EPOCHS = 150`, `TRAIN_SCANS = 0`, `RUN_NAME = "full"`).
Click "Save Version", choose "Save & Run All (Commit)", and click "Save".
Kaggle then runs the whole notebook on its own computers. You can close the browser and the laptop.

**To see the results:** open this notebook's page on Kaggle and open the "Output" tab of the newest version.
The files are in `runs/RUN_NAME/`: `best.pt` and `last.pt` (the trained model), `log.csv` (numbers for every epoch), and pictures.

**If a run stops before the end:** run again with the same `RUN_NAME`. The training continues after the last saved epoch.
This works because this notebook's own output is added as an input (see the "Input" panel).
"""

K_SETTINGS = """EPOCHS = 150          # how many passes over the training scans. 150 is the full run. Use 1 for a short test.
TRAIN_SCANS = 0       # 0 = use all training scans (the full run). Use 2 for a short test.
RUN_NAME = "full"     # name of the results folder. Use "test" for a short test."""

K_DATA = """import glob, os, shutil
files = sorted(glob.glob("/kaggle/input/**/*.npz", recursive=True))
assert files, "No prepared scans were found. Click 'Add Input' and add the dataset xray-prepared-scans."
os.makedirs("/tmp/prepared", exist_ok=True)
for f in files:
    shutil.copy(f, "/tmp/prepared/")
print(len(files), "prepared scans copied")

OUT = "/kaggle/working/runs/" + RUN_NAME
# Saved work of an earlier run. It is only found if that run's output was added to this notebook as an input.
earlier = sorted(glob.glob("/kaggle/input/**/runs/" + RUN_NAME + "/last.pt", recursive=True))
if earlier and not os.path.exists(OUT + "/last.pt"):
    shutil.copytree(os.path.dirname(earlier[-1]), OUT, dirs_exist_ok=True)
    print("copied the saved work of an earlier run:", earlier[-1])
else:
    print("no saved work of an earlier run was found")"""

K_FOLDERS = """os.makedirs("/kaggle/working/project/src", exist_ok=True)
os.makedirs("/kaggle/working/project/scripts", exist_ok=True)"""

K_TRAIN = """!python /kaggle/working/project/scripts/train.py --data /tmp/prepared --out "$OUT" --epochs $EPOCHS --limit $TRAIN_SCANS"""


def build_kaggle():
    cells = [
        markdown(K_INTRO),
        code(K_SETTINGS),
        code(GPU_CHECK.replace("Choose Runtime > Change runtime type > T4 GPU, then run again.",
                               "In the right panel choose Session options > Accelerator > GPU T4 x2, then run again.")),
        code(K_DATA),
        code(K_FOLDERS),
        markdown("The next 7 cells write the project's code files into Kaggle. "
                 "They are copies of the files in the project folder."),
    ]
    cells += [writefile_cell(p, "/kaggle/working/project") for p in EMBEDDED_FILES]
    cells += [
        markdown("The next cell trains the model. After every epoch it saves the model in the results folder."),
        code(K_TRAIN),
        code(RESULTS),
    ]
    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    path = KAGGLE_DIR / "xray_to_3d_kaggle.ipynb"
    notebook = {"cells": cells, "metadata": {k: v for k, v in METADATA.items() if k != "colab"}, "nbformat": 4, "nbformat_minor": 0}
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    meta = {
        "id": f"{KAGGLE_USER}/{KAGGLE_NOTEBOOK}",
        "title": KAGGLE_NOTEBOOK,
        "code_file": path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": [f"{KAGGLE_USER}/{KAGGLE_DATASET}"],
        "kernel_sources": [],
        "competition_sources": [],
    }
    meta_path = KAGGLE_DIR / "kernel-metadata.json"
    if meta_path.exists():  # keep inputs that were added by hand, for example an earlier run's output
        meta["kernel_sources"] = json.loads(meta_path.read_text()).get("kernel_sources", [])
    meta_path.write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {path} ({path.stat().st_size} bytes, {len(cells)} cells)")


# ---------------------------------------------------------------- Kaggle, second notebook: scoring
# This notebook does NOT train. It loads the finished model from the output of the training
# notebook, runs it on scans it was not trained on, and measures the result in millimeters.
KAGGLE_SCORE_NOTEBOOK = "xray-to-3d-scoring"
KAGGLE_SCORE_DIR = ROOT / "notebooks" / "kaggle_scoring"
SCORE_FILES = [
    "src/fake_panoramic.py",
    "src/curved_volume.py",
    "src/model.py",
    "src/train_data.py",
    "scripts/predict.py",
    "scripts/measure.py",
]

S_INTRO = f"""
# Panoramic X-ray to 3D scan: scoring the trained model (Kaggle)

This notebook does not train anything. It takes about 5 minutes.

1. It loads the trained model (`best.pt`) from the output of the notebook `{KAGGLE_NOTEBOOK}`.
2. It runs the model on the 5 check scans. Their scores must be the same as the scores the training printed. This shows the model file was loaded correctly.
3. It runs the model on the 10 test scans. These scans were never used during training.
4. It measures how far the model's output is from the real 3D scan, in millimeters.

**To see the results:** open the "Output" tab. `results/test/results.csv` holds the numbers for each scan,
`results/test/overview_NUMBER.png` and `results/test/cross_NUMBER.png` are pictures, and
`predictions/test/NUMBER.npz` is the model's output for each scan.
"""

S_DATA = """import glob, os, shutil
files = sorted(glob.glob("/kaggle/input/**/*.npz", recursive=True))
assert files, "No prepared scans were found. Click 'Add Input' and add the dataset xray-prepared-scans."
os.makedirs("/tmp/prepared", exist_ok=True)
for f in files:
    shutil.copy(f, "/tmp/prepared/")
print(len(files), "prepared scans copied")

models = sorted(glob.glob("/kaggle/input/**/runs/full/best.pt", recursive=True))
assert models, "The trained model was not found. Click 'Add Input' and add the output of the notebook xray-to-3d-training."
MODEL = models[-1]
print("model file:", MODEL, round(os.path.getsize(MODEL) / 1e6), "MB")"""

S_CHECK = """!python /kaggle/working/project/scripts/predict.py --data /tmp/prepared --model "$MODEL" --out /tmp/predictions/check --group check
!python /kaggle/working/project/scripts/measure.py --data /tmp/prepared --pred /tmp/predictions/check --out /kaggle/working/results/check --no-pictures"""

S_COMPARE = """import json, pandas as pd
log = pd.read_csv(os.path.dirname(MODEL) + "/log.csv")
epoch = json.load(open("/tmp/predictions/check/about.json"))["epoch"]
during_training = log[log.epoch == epoch].iloc[0]
now = pd.read_csv("/kaggle/working/results/check/results.csv").iloc[-1]
names = ["err_hu", "psnr", "dice lower jaw", "dice upper jaw", "dice nerve canal", "dice sinus", "dice teeth"]
table = pd.DataFrame({"during training": [during_training[n] for n in names], "now": [float(now[n]) for n in names]}, index=names)
display(table.round(3))
worst = (table["during training"] - table["now"]).abs().div(table["during training"].abs()).max()
print("SAME AS DURING TRAINING" if worst < 0.01 else "DIFFERENT FROM TRAINING: the model file may not be loaded correctly")"""

S_TEST = """!python /kaggle/working/project/scripts/predict.py --data /tmp/prepared --model "$MODEL" --out /kaggle/working/predictions/test --group test
!python /kaggle/working/project/scripts/measure.py --data /tmp/prepared --pred /kaggle/working/predictions/test --out /kaggle/working/results/test"""

S_SHOW = """from IPython.display import Image, display
display(pd.read_csv("/kaggle/working/results/test/results.csv").round(2).T)
first = sorted(glob.glob("/kaggle/working/results/test/overview_*.png"))[0]
display(Image(first, width=1100))
display(Image(first.replace("overview_", "cross_"), width=1100))"""


def build_kaggle_scoring():
    cells = [
        markdown(S_INTRO),
        code(GPU_CHECK.replace("Choose Runtime > Change runtime type > T4 GPU, then run again.",
                               "In the right panel choose Session options > Accelerator > GPU T4 x2, then run again.")),
        code(S_DATA),
        code(K_FOLDERS),
        markdown(f"The next {len(SCORE_FILES)} cells write the project's code files into Kaggle. "
                 "They are copies of the files in the project folder."),
    ]
    cells += [writefile_cell(p, "/kaggle/working/project") for p in SCORE_FILES]
    cells += [
        markdown("The next two cells run the model on the 5 check scans and compare the scores with the scores from training."),
        code(S_CHECK),
        code(S_COMPARE),
        markdown("The next cell runs the model on the 10 test scans and measures the result in millimeters."),
        code(S_TEST),
        code(S_SHOW),
    ]
    KAGGLE_SCORE_DIR.mkdir(parents=True, exist_ok=True)
    path = KAGGLE_SCORE_DIR / "xray_to_3d_scoring.ipynb"
    notebook = {"cells": cells, "metadata": {k: v for k, v in METADATA.items() if k != "colab"}, "nbformat": 4, "nbformat_minor": 0}
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    meta = {
        "id": f"{KAGGLE_USER}/{KAGGLE_SCORE_NOTEBOOK}",
        "title": KAGGLE_SCORE_NOTEBOOK,
        "code_file": path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": [f"{KAGGLE_USER}/{KAGGLE_DATASET}"],
        "kernel_sources": [f"{KAGGLE_USER}/{KAGGLE_NOTEBOOK}"],
        "competition_sources": [],
    }
    (KAGGLE_SCORE_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {path} ({path.stat().st_size} bytes, {len(cells)} cells)")


# ---------------------------------------------------------------- Kaggle, third notebook: Patient 1
# This notebook does NOT train. It runs the finished model on Patient 1's real X-ray.
# Its input is the private dataset patient1-xray, which holds only the small X-ray pictures made by
# scripts/place_xray.py (240 x 544 pixels, no name, ID or birth date).
KAGGLE_P1_NOTEBOOK = "xray-to-3d-patient1"
KAGGLE_P1_DATASET = "patient1-xray"
KAGGLE_P1_DIR = ROOT / "notebooks" / "kaggle_patient1"
P1_FILES = ["src/model.py", "src/train_data.py", "scripts/predict.py"]

P1_INTRO = f"""
# Panoramic X-ray to 3D scan: Patient 1 (Kaggle)

This notebook does not train anything. It takes about 3 minutes.
It loads the trained model (`best.pt`) from the output of the notebook `{KAGGLE_NOTEBOOK}` and runs it
on Patient 1's real panoramic X-ray (private dataset `{KAGGLE_P1_DATASET}`).
The model's outputs are saved in `predictions/patient1/` (see the "Output" tab).
The comparison with Patient 1's real 3D scan is done on the Mac, because the real 3D scan must not leave the Mac.
"""

P1_DATA = """import glob, os, shutil
files = sorted(glob.glob("/kaggle/input/**/p1_*.npz", recursive=True))
assert files, "Patient 1's X-ray was not found. Click 'Add Input' and add the dataset patient1-xray."
os.makedirs("/tmp/p1", exist_ok=True)
for f in files:
    shutil.copy(f, "/tmp/p1/")
NAMES = " ".join(os.path.basename(f)[:-4] for f in files)
print("X-ray versions:", NAMES)

models = sorted(glob.glob("/kaggle/input/**/runs/full/best.pt", recursive=True))
assert models, "The trained model was not found. Click 'Add Input' and add the output of the notebook xray-to-3d-training."
MODEL = models[-1]
print("model file:", MODEL, round(os.path.getsize(MODEL) / 1e6), "MB")"""

P1_RUN = """!python /kaggle/working/project/scripts/predict.py --data /tmp/p1 --model "$MODEL" --out /kaggle/working/predictions/patient1 --scans $NAMES"""


def build_kaggle_patient1():
    cells = [
        markdown(P1_INTRO),
        code(GPU_CHECK.replace("Choose Runtime > Change runtime type > T4 GPU, then run again.",
                               "In the right panel choose Session options > Accelerator > GPU T4 x2, then run again.")),
        code(P1_DATA),
        code(K_FOLDERS),
        markdown(f"The next {len(P1_FILES)} cells write the project's code files into Kaggle. "
                 "They are copies of the files in the project folder."),
    ]
    cells += [writefile_cell(p, "/kaggle/working/project") for p in P1_FILES]
    cells += [markdown("The next cell runs the model on each version of Patient 1's X-ray."), code(P1_RUN)]
    KAGGLE_P1_DIR.mkdir(parents=True, exist_ok=True)
    path = KAGGLE_P1_DIR / "xray_to_3d_patient1.ipynb"
    notebook = {"cells": cells, "metadata": {k: v for k, v in METADATA.items() if k != "colab"}, "nbformat": 4, "nbformat_minor": 0}
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    meta = {
        "id": f"{KAGGLE_USER}/{KAGGLE_P1_NOTEBOOK}",
        "title": KAGGLE_P1_NOTEBOOK,
        "code_file": path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": [f"{KAGGLE_USER}/{KAGGLE_P1_DATASET}"],
        "kernel_sources": [f"{KAGGLE_USER}/{KAGGLE_NOTEBOOK}"],
        "competition_sources": [],
    }
    (KAGGLE_P1_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {path} ({path.stat().st_size} bytes, {len(cells)} cells)")


def main():
    build_kaggle()
    build_kaggle_scoring()
    build_kaggle_patient1()
    cells = [
        markdown(INTRO),
        code(SETTINGS),
        code(GPU_CHECK),
        code(DRIVE),
        code(FOLDERS),
        markdown("The next 7 cells write the project's code files into Colab. "
                 "They are copies of the files in the project folder."),
    ]
    cells += [writefile_cell(p) for p in EMBEDDED_FILES]
    cells += [
        markdown("The next cell makes a fake X-ray and a straightened 3D block from each scan, "
                 "and saves overview pictures. The first time it takes about 10 minutes and saves a copy "
                 "(about 570 MB) to Google Drive. After that it uses the copy and takes about 1 minute."),
        code(PREPARE),
        code(PREVIEW),
        markdown("The next cell trains the model and saves the results to Google Drive after every epoch. "
                 "To pause, click its stop button. To continue, run it again."),
        code(TRAIN),
        code(RESULTS),
    ]
    notebook = {"cells": cells, "metadata": METADATA, "nbformat": 4, "nbformat_minor": 0}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT_PATH} ({OUT_PATH.stat().st_size} bytes, {len(cells)} cells)")


if __name__ == "__main__":
    main()
