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
_DEFAULT_DOGS = frozenset({
    # tiny seed so code still works if file missing
    "labrador retriever", "golden retriever", "german shepherd",
    "bulldog", "poodle", "beagle", "rottweiler", "boxer",
    "chihuahua", "siberian husky", "dachshund", "great dane", "border collie"
})

_DEFAULT_DOG_STEMS = frozenset({
    # broad “family” words that appear in many breed names
    "terrier", "retriever", "hound", "spaniel", "shepherd",
    "pointer", "setter", "mastiff", "collie", "poodle", "bulldog"
})

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
                label = _canon(line)
                if label:  # skip empty after normalization
                    s.add(label)

    # Make sure stems are present (harmless if already there)
    s.update(_DEFAULT_DOG_STEMS)

    # Return immutable to avoid accidental mutation of the cached object
    return frozenset(s)

def is_dog_label(label: str, dognames_path: str = "data/dognames.txt") -> bool:
    
    # label_c = _canon(label)    
    label_c = _canon(label)

    # dogs = _dog_set(dognames_path)
    dogs = _dog_set(dognames_path)
    
    # # # return True on exact match OR if any word of label_c is in dogs
    # # if label_c in dogs:
    # #      return True
    
    # # words = label_c.split()
    
    # # for w in words:
    # #      if w in dogs:
    # #          return True
         
    # return False


    
dogs = _dog_set("C:/Users/amir_/OneDrive/Documentos/MEGA/Etudes/programming/python/ai-programming-with-python/project/dog-breed-pretrained/data/dognames.txt")

print(f"Total normalized entries: {len(dogs)}")
for i, name in enumerate(sorted(dogs)[:40], 1):
    print(f"{i:>3}. {name}")






