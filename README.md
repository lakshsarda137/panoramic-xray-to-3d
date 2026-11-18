# Panoramic X-ray to 3D CBCT

A model that takes one panoramic dental X-ray and outputs a 3D CBCT-like scan of the jaw, with the lower jaw, upper jaw, inferior alveolar nerve canal, maxillary sinus and teeth marked.

## Method

- **Training pairs.** No public dataset has real panoramic X-rays paired with CBCT scans. So a panoramic X-ray is computed from each ToothFairy3 CBCT scan (`src/fake_panoramic.py`), and the CBCT scan is the target.
- **Output format.** The model outputs the CBCT resampled along the dental arch (a "straightened" volume: height × position along the arch × depth across the arch), the same idea as Oral-3D. The output is bent back along an arch curve to give a normal 3D scan (`src/curved_volume.py`).
- **Model.** A 2D U-Net followed by a small 3D head (`src/model.py`). It outputs brightness (Hounsfield units) and 6 classes.
- **Data split.** 46 training scans, 5 validation scans, 10 test scans (ToothFairy3 "F" scans, 0.4 mm voxels).

## Code

| Step | File |
|---|---|
| Prepare training scans (computed X-ray + straightened volume + labels) | `scripts/prepare_tf3.py` |
| Train (run on Kaggle or Colab, needs a GPU) | `scripts/train.py`, notebooks in `notebooks/` |
| Predict | `scripts/predict.py` |
| Measure in millimetres (canal position, sinus floor, bone thickness, bone density) | `scripts/measure.py` |
| Patient 1: place the real X-ray, build the 3D scan, compare with the real CBCT | `scripts/place_xray.py`, `scripts/compare_patient1.py`, `scripts/compare_fake_patient1.py` |

`scripts/build_colab_notebook.py` builds the Kaggle and Colab notebooks from `src/` and `scripts/`.

## Data

Not included. Patient data is kept out of this repository. The training data is the ToothFairy3 dataset (CC BY-NC-SA), available at https://ditto.ing.unimore.it/toothfairy3/.
