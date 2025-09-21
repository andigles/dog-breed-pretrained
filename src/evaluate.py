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
