"""
Feature selection pipeline — task-oriented package layout.

Public API for downstream modeling code:
"""
from src.selection.build_dataset import build_task1_dataset
from src.selection.shared.manifest import load_selection_manifest

__all__ = [
    "build_task1_dataset",
    "load_selection_manifest",
]
