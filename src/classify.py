"""
Reusable inference for pretrained CNNs (AlexNet/VGG16/ResNet18).

- Loads model with TorchVision *weights enums* (no external txt mapping).
- Normalizes labels (lowercase, underscores/hyphens → spaces).
- Provides a dog/not-dog helper using data/dognames.txt (with fallback).
"""

from __future__ import annotations

# Standard lib
import os
import re
from functools import lru_cache
from typing import Tuple

# Third-party
import torch
from PIL import Image, ImageFile
from torchvision import models
from torchvision.models import ResNet18_Weights, VGG16_Weights, AlexNet_Weights

# In case a sample image is slightly corrupted:
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Module constants (private)
# Tiny, resilient fallback if the file path is wrong or missing.
_DEFAULT_DOGS = frozenset({
    "labrador retriever",
    "golden retriever",
    "german shepherd",
    "poodle",
    "bulldog",
    "beagle",
    "rottweiler",
    "yorkshire terrier",
    "boxer",
    "dachshund",
    "siberian husky",
    "great dane",
    "chihuahua",
    "shih tzu",
    "border collie",
    "french bulldog",
    "doberman pinscher",
    "australian shepherd",
    "cocker spaniel",
    "saint bernard",
})

# Curated “family” tokens that safely boost recall.
# Keep them fairly specific to avoid false positives.
_DEFAULT_DOG_STEMS = frozenset({
    "terrier",
    "retriever",
    "hound",
    "spaniel",
    "shepherd",
    "sheepdog",
    "pointer",
    "setter",
    "mastiff",
    "collie",
    "poodle",
    "bulldog",
    "schnauzer",
    "pinscher",
    "spitz",
    "ridgeback",
    "corgi",
})

class _ModelBundle:
    def __init__(self, model, preprocess, categories):
        self.model = model
        self.preprocess = preprocess
        self.categories = categories


def _canon(s: str) -> str:
    """
    Lowercase, trim, replace '_' and '-' with spaces, drop other punctuation,
    and collapse inner whitespace.
    """
    s = s.lower()
    s = re.sub(r"[_-]", " ", s)          # underscores/hyphens → space
    s = re.sub(r"[^a-z\s]", " ", s)      # any other punctuation → space
    s = re.sub(r"\s+", " ", s).strip()   # collapse + full trim

    return s

@lru_cache(maxsize=1)
def _dog_set(path: str = "data/dognames.txt") -> frozenset[str]:
    """
    Build the canonical dog label set.

    Order of precedence:
      1) Start with built-in defaults (_DEFAULT_DOGS)
      2) If 'path' exists, add normalized lines from that file
      3) Ensure stems like 'terrier', 'retriever', etc. are included

    Returns:
        frozenset[str]: immutable set of normalized dog tokens/labels.
    """
    s: set[str] = set(_DEFAULT_DOGS)

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                for alias in line.split(","):
                    label = _canon(alias)
                    if label:
                        s.add(label)

    # Make sure stems are present (harmless if already there)
    s.update(_DEFAULT_DOG_STEMS)

    # Return immutable to avoid accidental mutation of the cached object
    return frozenset(s)

def is_dog_label(label: str, dognames_path: str = "data/dognames.txt") -> bool:
    dogs = _dog_set(dognames_path)

    # IMPORTANT: check each comma-separated piece from the classifier
    for piece in label.split(","):
        chunk = _canon(piece)
        if not chunk:
            continue

        # 1) Exact match still helps for simple cases
        if chunk in dogs:

            return True

        # 2) Phrase containment: does any alias appear in the chunk?
        # Use word boundaries to avoid partial-token hits (e.g., "hound" in "houndstooth")
        for alias in dogs:
            if re.search(rf"\b{re.escape(alias)}\b", chunk):
                
                return True
            
    return False

@lru_cache(maxsize=3)
def load_model(name: str = "resnet") -> _ModelBundle:
    # pick weights & model based on name
    # call model.eval()
    # get preprocess and categories
    # sanity-check categories not None
    # return _ModelBundle(model, preprocess, categories)
    
    name = name.lower()

    if name == "resnet":
        weights = ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    elif name == "vgg":
        weights = VGG16_Weights.DEFAULT
        model = models.vgg16(weights=weights)
    elif name == "alexnet":
        weights = AlexNet_Weights.DEFAULT
        model = models.alexnet(weights=weights)
    else:    
        raise ValueError(f"Unsupported model name: {name}. Supported: resnet, vgg, alexnet")
    
    model.eval()
    
    preprocess = weights.transforms()
    categories = weights.meta.get("categories", None)

    if categories is None:
        raise ValueError(f"Model weights for {name} do not include categories.")

    return _ModelBundle(model, preprocess, categories)