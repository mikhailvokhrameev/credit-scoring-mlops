"""Static checks that keep the notebooks from drifting away from `src/` (nothing is executed)."""
import ast
import importlib.util
from importlib import import_module
from pathlib import Path

import nbformat
import pytest

NOTEBOOKS = sorted(Path(__file__).resolve().parents[1].joinpath("notebooks").glob("*.ipynb"))
assert NOTEBOOKS, "no notebooks found in notebooks/"


def _code_cells(nb_path: Path) -> list[str]:
    nb = nbformat.read(nb_path, as_version=4)
    return [c.source for c in nb.cells if c.cell_type == "code"]


def _python_source(cell: str) -> str:
    """Drop IPython shell (`!`) and magic (`%`) lines, which `ast` cannot parse."""
    return "\n".join(line for line in cell.splitlines() if not line.lstrip().startswith(("!", "%")))


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=lambda p: p.name)
def test_code_cells_parse(nb_path):
    for i, cell in enumerate(_code_cells(nb_path)):
        try:
            ast.parse(_python_source(cell))
        except SyntaxError as e:
            pytest.fail(f"{nb_path.name}, code cell {i}: {e}")


def _is_src(name: str | None) -> bool:
    return name == "src" or (name or "").startswith("src.")


def _has_submodule(package: str, name: str) -> bool:
    try:
        return importlib.util.find_spec(f"{package}.{name}") is not None
    except ModuleNotFoundError:  # `package` is a plain module, not a package
        return False


def _unresolved_src_imports(source: str) -> list[str]:
    """Every `src` import in `source` (any form) that does not resolve, as readable strings."""
    errors = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_src(alias.name):
                    try:
                        import_module(alias.name)
                    except ModuleNotFoundError:
                        errors.append(f"`import {alias.name}` does not resolve")
        elif isinstance(node, ast.ImportFrom) and _is_src(node.module):
            try:
                module = import_module(node.module)
            except ModuleNotFoundError:
                errors.append(f"`from {node.module} import ...` does not resolve")
                continue
            for alias in node.names:
                if not hasattr(module, alias.name) and not _has_submodule(node.module, alias.name):
                    errors.append(f"`from {node.module} import {alias.name}` does not resolve")
    return errors


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=lambda p: p.name)
def test_src_imports_resolve(nb_path):
    for i, cell in enumerate(_code_cells(nb_path)):
        errors = _unresolved_src_imports(_python_source(cell))
        assert not errors, f"{nb_path.name}, code cell {i}: {errors}"


@pytest.mark.parametrize(
    "source, ok",
    [
        ("import src.config", True),
        ("from src import config", True),
        ("from src.config import load_params", True),
        ("from src.models import train", True),
        ("import src.nope", False),
        ("from src import nope", False),
        ("from src.config import nope", False),
    ],
)
def test_unresolved_src_imports_detects_all_forms(source, ok):
    assert (_unresolved_src_imports(source) == []) is ok


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=lambda p: p.name)
def test_outputs_are_cleared(nb_path):
    for i, cell in enumerate(nbformat.read(nb_path, as_version=4).cells):
        if cell.cell_type == "code":
            assert cell.outputs == [] and cell.execution_count is None, (
                f"{nb_path.name}, cell {i}: commit notebooks without outputs"
            )
