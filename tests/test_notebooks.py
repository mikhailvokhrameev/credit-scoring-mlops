"""Static checks that keep the notebooks from drifting away from `src/` (nothing is executed)."""
import ast
from importlib import import_module
from pathlib import Path

import nbformat
import pytest

NOTEBOOKS = sorted(Path(__file__).resolve().parents[1].joinpath("notebooks").glob("*.ipynb"))


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


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=lambda p: p.name)
def test_src_imports_resolve(nb_path):
    for i, cell in enumerate(_code_cells(nb_path)):
        for node in ast.walk(ast.parse(_python_source(cell))):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("src."):
                module = import_module(node.module)
                for alias in node.names:
                    assert hasattr(module, alias.name), (
                        f"{nb_path.name}, code cell {i}: `from {node.module} import {alias.name}` does not resolve"
                    )


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=lambda p: p.name)
def test_outputs_are_cleared(nb_path):
    for i, cell in enumerate(nbformat.read(nb_path, as_version=4).cells):
        if cell.cell_type == "code":
            assert cell.outputs == [] and cell.execution_count is None, (
                f"{nb_path.name}, cell {i}: commit notebooks without outputs"
            )
