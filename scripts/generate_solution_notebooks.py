#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHALLENGES_DIR = ROOT / "challenges"
NOTEBOOKS_DIR = ROOT / "notebooks"
FRAMEWORKS = ("pytorch", "jax", "triton", "mlx")


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def markdown_cell(text: str):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code_cell(text: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def solution_path(challenge_dir: Path, framework: str) -> Path | None:
    path = challenge_dir / "solution" / f"solution.{framework}.py"
    return path if path.exists() else None


def challenge_notebook_cells(challenge_dir: Path):
    challenge_rel = challenge_dir.relative_to(ROOT)
    title = challenge_dir.name
    challenge_html = (challenge_dir / "challenge.html").read_text()
    cells = [
        markdown_cell(
            f"# {title}\n\n"
            f"Audience: junior researcher\n\n"
            f"- Challenge path: `{challenge_rel}`\n"
            f"- Source spec: [{challenge_rel / 'challenge.html'}](../challenge.html)\n"
            f"- Source implementation: [{challenge_rel / 'challenge.py'}](../challenge.py)\n"
        ),
        markdown_cell(
            "## Problem Statement\n\n"
            "The original challenge HTML is embedded below so the notebook stays close to the repo source.\n\n"
            f"{challenge_html}"
        ),
        markdown_cell(
            "## Framework Coverage\n\n"
            "This notebook collects the currently available solution artifacts for PyTorch, JAX, Triton, and MLX."
        ),
    ]
    for framework in FRAMEWORKS:
        path = solution_path(challenge_dir, framework)
        if path is None:
            cells.append(markdown_cell(f"## {framework.title()}\n\nNo solution file is present yet."))
            continue
        code = path.read_text()
        cells.append(markdown_cell(f"## {framework.title()}\n\nSource: `{path.relative_to(ROOT)}`"))
        cells.append(code_cell(code))
    cells.append(
        markdown_cell(
            "## Verification Notes\n\n"
            "Use `python scripts/verify_matrix_solutions.py` for the local matrix-operation verifier.\n"
            "GPU-only Triton validation still depends on a remote NVIDIA environment."
        )
    )
    return cells


def write_notebook(path: Path, cells):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook(cells), indent=2) + "\n")


def build_index():
    lines = [
        "# LeetGPU Solution Index",
        "",
        "Audience: junior researcher",
        "",
        "This index links every challenge notebook and shows which framework solution files exist.",
        "",
    ]
    for difficulty in ("easy", "medium", "hard"):
        lines.append(f"## {difficulty.title()}")
        lines.append("")
        for challenge_dir in sorted((CHALLENGES_DIR / difficulty).iterdir()):
            if not challenge_dir.is_dir():
                continue
            coverage = []
            for framework in FRAMEWORKS:
                if solution_path(challenge_dir, framework) is not None:
                    coverage.append(framework)
            coverage_text = ", ".join(coverage) if coverage else "no framework solutions yet"
            notebook_rel = challenge_dir.relative_to(ROOT) / "notebook" / "explanation.ipynb"
            lines.append(f"- [{challenge_dir.name}]({Path('..') / notebook_rel}): {coverage_text}")
        lines.append("")
    return [markdown_cell("\n".join(lines) + "\n")]


def main() -> int:
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    for difficulty in ("easy", "medium", "hard"):
        for challenge_dir in sorted((CHALLENGES_DIR / difficulty).iterdir()):
            if not challenge_dir.is_dir():
                continue
            write_notebook(challenge_dir / "notebook" / "explanation.ipynb", challenge_notebook_cells(challenge_dir))
    write_notebook(NOTEBOOKS_DIR / "index.ipynb", build_index())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
