"""Architecture and import-boundary validation."""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

SRC_ROOT = Path("src")


def _module_paths(package: str) -> list[Path]:
    root = SRC_ROOT / package.replace(".", "/")
    return sorted(root.rglob("*.py"))


def _imports_from_pipelines(module_path: Path) -> list[str]:
    tree = ast.parse(module_path.read_text())
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("src.pipelines"):
                imports.append(node.module)
    return imports


@pytest.mark.parametrize(
    "module_path",
    _module_paths("evaluation"),
    ids=lambda p: str(p.relative_to(SRC_ROOT)),
)
def test_evaluation_does_not_import_pipelines(module_path: Path) -> None:
    assert not _imports_from_pipelines(module_path), (
        f"{module_path} must not import pipelines: "
        f"{_imports_from_pipelines(module_path)}"
    )


@pytest.mark.parametrize(
    "module_path",
    _module_paths("models"),
    ids=lambda p: str(p.relative_to(SRC_ROOT)),
)
def test_models_do_not_import_evaluation(module_path: Path) -> None:
    tree = ast.parse(module_path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("src.evaluation"), (
                f"{module_path} must not import evaluation"
            )


@pytest.mark.parametrize(
    "module_path",
    _module_paths("inference"),
    ids=lambda p: str(p.relative_to(SRC_ROOT)),
)
def test_inference_does_not_import_pipelines(module_path: Path) -> None:
    assert not _imports_from_pipelines(module_path), (
        f"{module_path} must not import pipelines: "
        f"{_imports_from_pipelines(module_path)}"
    )


def test_play_type_train_delegates_cv_to_evaluation() -> None:
    from src.pipelines.play_type import train as play_train

    source = inspect.getsource(play_train)
    assert "cross_validate_classifier" in source
    assert "StratifiedKFold" not in source
