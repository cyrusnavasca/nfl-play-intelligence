"""
Feature selection pipeline — task-oriented package layout.

Public API for downstream modeling code:
"""
from src.selection.play_type.build_dataset import build_task1_dataset
from src.selection.shared.manifest import load_selection_manifest
from src.selection.yards_gained.build_dataset import build_task2_dataset

__all__ = [
    "build_task1_dataset",
    "build_task2_dataset",
    "load_selection_manifest",
]
