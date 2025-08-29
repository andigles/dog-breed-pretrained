# Dog Breed — Pretrained Baselines

**Scope:** Compare pretrained CNN (convolutional neural network) models — AlexNet / VGG / ResNet — on a tiny dog‑vs‑not‑dog image set and print overall scores.  
**Runs in minutes:** uses a few sample images; **no training** required.

> Status: The original Udacity driver (`check_images.py`) runs and is kept for transparency.  
> Next step: a small modern CLI in `src/` that prints a summary **and** writes `results/results.csv`.

---

## Quickstart

### Option A — Conda (recommended if you like an `environment.yml`)

```bash
# From the repo root
conda env create -f environment.yml
conda activate dog-breed-pretrained

# Run the legacy driver on the tiny sample set
python check_images.py --dir data/sample/pet_images --arch vgg --dogfile data/dognames.txt
```

### Option B — Python venv + pip

```bash
python -m venv .venv
# Windows (Git Bash):
source .venv/Scripts/activate
pip install -r requirements.txt

python check_images.py --dir data/sample/pet_images --arch vgg --dogfile data/dognames.txt
```

### Notes

- On first run, PyTorch may download model weights to your user cache (outside the repo).  
- You might see deprecation warnings about `pretrained` vs `weights` in TorchVision — harmless for the legacy script.
- Large datasets should live under `data/raw/` (already git‑ignored). The repo only keeps a tiny demo set in `data/sample/`.

---

## What’s in this repo

```markdownlint
dog-breed-pretrained/
├─ README.md
├─ requirements.txt
├─ environment.yml
├─ check_images.py              # legacy Udacity driver (CLI)
├─ classifier.py                # legacy classifier helper (modernized soon)
├─ classify_images.py           # legacy loop (patched to fix path join)
├─ data/
│  ├─ dognames.txt
│  ├─ sample/
│  │  ├─ pet_images/           # 2–3 dog + 2–3 not-dog images
│  │  └─ uploaded_images/      # 1–2 extra demo images
│  └─ raw/                      # place for any large/untracked image dumps
├─ results/                     # small, git-tracked outputs (CSV once src/ is added)
├─ .gitignore
└─ .gitattributes
```

---

## Usage (legacy driver)

The legacy script classifies images in a folder using a chosen pretrained model and compares predictions to labels inferred from filenames.

```bash
python check_images.py --dir data/sample/pet_images --arch resnet --dogfile data/dognames.txt
# or
python check_images.py --dir data/sample/pet_images --arch vgg --dogfile data/dognames.txt
# or
python check_images.py --dir data/sample/pet_images --arch alexnet --dogfile data/dognames.txt
```

**Tip:** Filenames are treated as the ground‑truth label after lowercasing and cleaning underscores/hyphens (e.g., `Golden_retriever_05182.jpg` → `golden retriever`).

---

## Roadmap (next commits)

- Add `src/classify.py` (reusable inference with TorchVision `weights.meta['categories']`).
- Add `src/evaluate.py` (simple CLI; prints per‑model summary and writes `results/results.csv`).
- Update README with the new “Run in one command” example for `src/`.
- Optional: add small tests in `tests/` for label cleanup and dog/not‑dog checks.
- Optional: add MIT License.

---

## Credits

Adapted from the Udacity **AI Programming with Python** project “Use a Pre‑trained Image Classifier to Identify Dog Breeds.”
