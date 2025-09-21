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

