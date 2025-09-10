# src/evaluate.py
from __future__ import annotations

from dataclasses import dataclass 
from pathlib import Path 
import argparse 
import csv 
import re 
from typing import Iterable 

from .classify import classify_image, is_dog_label, _canon 