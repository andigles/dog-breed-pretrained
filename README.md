# Dog Breed — Pretrained Baselines

Evaluate pretrained convolutional neural network (CNN) models (AlexNet / VGG / ResNet) on a folder of images.  
Runs in minutes — no training — prints a per‑model summary and writes a CSV with detailed results.

> **CLI** = Command‑Line Interface

---

## Quickstart (new CLI)

### Requirements

- Python 3.10+ (adjust if your environment differs)
- PyTorch and TorchVision versions compatible with your Python (installed via `requirements.txt`)

### Install

```bash
git clone https://github.com/andigles/dog-breed-pretrained.git
cd dog-breed-pretrained

# Option A) Virtual environment + pip (recommended for simplicity)
python -m venv .venv
# Windows (Git Bash):
source .venv/Scripts/activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt

# Option B) Conda (only if you prefer conda)
# conda env create -f environment.yml
# conda activate dog-breed-pretrained
```

### Run (evaluate all models)

```bash
python -m src.evaluate --images data/sample/pet_images --model all --out results/results.csv
```

### Run (single model example)

```bash
python -m src.evaluate --images data/sample/pet_images --model resnet
```

### Outputs

- **Console**: For each model, prints Images count, Dog/not-dog accuracy, Label‑match rate.
- **File**: Creates `results/results.csv` with one row per image prediction.

---

## Project structure

```plaintext
.
├─ src/
│  ├─ classify.py          # single‑image inference helper
│  └─ evaluate.py          # CLI: walks folder, runs models, writes CSV and prints summary
├─ data/
│  ├─ dognames.txt         # list of dog breeds used for is_dog checks
│  └─ sample/pet_images/   # tiny demo set (you can point to your own images)
├─ results/                # created on first run (CSV lives here)
├─ README.md
├─ requirements.txt        # Python dependencies
└─ environment.yml         # conda environment (optional)

---

## Troubleshooting

- **Relative import error ("attempted relative import with no known parent package")**  
  Run from the repository root using **module mode**:  
  `python -m src.evaluate --images ...`

- **No output printed**  
  Make sure `--images` points to a folder that actually contains images (`.jpg/.jpeg/.png`).  
  Try `ls data/sample/pet_images` (macOS/Linux) or `dir data\sample\pet_images` (Windows) to verify.

- **`results/` not found**  
  The CLI creates the folder on write. If it does not, check file permissions in the repo directory.

- **TorchVision warnings about `pretrained` vs `weights`**  
  They are harmless in this project. We only perform inference; follow the warning’s guidance if you pin newer versions.

---

## Development notes

- The evaluator walks a folder of images and runs **one** model (e.g., `resnet`) or **all** supported models (`--model all`).  
- It prints a per‑model summary and writes a CSV with columns:
  - `image, model, truth_label, pred_label, truth_is_dog, pred_is_dog, label_match, conf`
- Truth labels are inferred from file names (digits removed, underscores/hyphens normalized, then canonicalized).

---

## Legacy (reference)

Kept for historical context from the Udacity project; **prefer the new CLI above**.

### Legacy usage

```bash
python legacy/check_images.py --dir data/sample/pet_images --arch resnet --dogfile data/dognames.txt
python legacy/check_images.py --dir data/sample/pet_images --arch vgg --dogfile data/dognames.txt
python legacy/check_images.py --dir data/sample/pet_images --arch alexnet --dogfile data/dognames.txt
```

Notes:

- Paths and option names in the legacy script differ from the new `src/` layout.
- The new CLI consolidates outputs and generates a reproducible CSV.

---

## Roadmap / Status

- ✅ `src/classify.py` — reusable inference (`classify_image`, `_canon`, `is_dog_label`).
- ✅ `src/evaluate.py` — CLI that prints summary and writes `results/results.csv`.
- ⏳ Light tests for label cleanup and dog/not‑dog checks (optional).
- ✅ MIT License file.

---

## Credits

Adapted from Udacity’s “Use a Pre‑trained Image Classifier to Identify Dog Breeds” and extended with a modern CLI and evaluation CSV.

License: MIT © 2025 Andrés Iglesias
