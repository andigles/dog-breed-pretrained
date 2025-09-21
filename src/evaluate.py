# src/evaluate.py
from __future__ import annotations

from dataclasses import dataclass 
from pathlib import Path 
import argparse 

import csv 
import re 
from typing import Iterable 
from .classify import classify_image, is_dog_label, _canon

_NUMS = re.compile(r"\d+")

@dataclass(frozen=True)
class ResultRow:
    image: str
    model: str
    truth_label: str
    pred_label: str
    truth_is_dog: bool
    pred_is_dog: bool
    label_match: bool
    conf: float

def list_images(root: Path, exts: set[str] = {".jpg", ".jpeg", ".png"}) -> list[Path]:
    root = Path(root)
    imgs = [p for p in root.iterdir() if p.suffix.lower() in exts]
    imgs.sort()
    return imgs

def truth_from_path(p: Path) -> str:
    stem = _NUMS.sub("", p.stem)
    return _canon(stem)

def eval_model_on_images(model_name: str, images: list[Path]) -> list[ResultRow]:
    rows: list[ResultRow] = []

    for img_path in images:
        truth = truth_from_path(img_path)
        truth_is_dog = is_dog_label(truth)
        
        pred_label, pred_is_dog, conf = classify_image(str(img_path), model_name)

        label_match = (_canon(pred_label) == truth)

        rows.append(
            ResultRow(
                image=str(img_path.name),
                model=model_name,
                truth_label=truth,
                pred_label=pred_label,
                truth_is_dog=truth_is_dog,
                pred_is_dog=pred_is_dog,
                label_match=label_match,
                conf=conf
            )
        )

    return rows

def summarize(rows: list[ResultRow]) -> None:
    by_model: dict[str, list[ResultRow]] = {}
    for r in rows:
        by_model.setdefault(r.model, []).append(r)

    for model, mrows in by_model.items():
        by_model: dict[str, list[ResultRow]] = {}
        n = len(mrows)
        dog_acc = (sum(r.truth_is_dog == r.pred_is_dog for r in mrows) / n) if n else 0.0
        label_rate = (sum(r.label_match for r in mrows) / n) if n else 0.0

        print(f"\nModel: {model}")
        print(f"  Images:             {n}")
        print(f"  Dog/not-dog accuracy: {dog_acc:.3f}")
        print(f"  Label-match rate:     {label_rate:.3f}")    

def write_csv(rows: list[ResultRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "model", "truth_label", "pred_label", "truth_is_dog", "pred_is_dog", "label_match", "conf"])
        for r in rows:
            w.writerow([
                r.image,
                r.model,
                r.truth_label,
                r.pred_label,
                int(r.truth_is_dog),
                int(r.pred_is_dog),
                int(r.label_match),
                f"{r.conf:.6f}"
            ])

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate image classifiers over a folder of images."
    )
    p.add_argument(
        "--images", required=True, help="Folder with images"
    )
    p.add_argument(
        "--model",
        default="resnet",
        choices=["resnet", "vgg", "alexnet", "all"],
        help="Which model to run (or 'all')",
    )
    p.add_argument(
        "--out",
        default="results/results.csv",
        help="CSV output path",
    )
    return p.parse_args()

def main() -> int:
    args = parse_args()
    images = list_images(Path(args.images))
    if not images: 
        print("No images found. Check --images path and extensions.") 
        return 2

    models = ["resnet", "vgg", "alexnet"] if args.model == "all" else [args.model]
    print("Models to run:", ", ".join(models))

    all_rows: list[ResultRow] = []
    for model in models:
        all_rows.extend(eval_model_on_images(model, images))
    
    summarize(all_rows)
    write_csv(all_rows, Path(args.out))
    print(f"\nWrote results to {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())




