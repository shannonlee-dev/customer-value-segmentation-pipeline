"""Validate clean notebook source or executed portable evidence."""

import argparse
import hashlib
from pathlib import Path
import nbformat

REQUIRED_MARKERS = ("Analysis scope: FULL DATASET", "Customer sampling: NONE", "Product sampling: NONE", "Image analysis sampling: NONE")


def inspect_notebook(path, mode="executed"):
    notebook = nbformat.read(Path(path), as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)
    if mode == "source":
        if any(cell.get("outputs") or cell.get("execution_count") is not None for cell in notebook.cells if cell.cell_type == "code"):
            raise ValueError("Source notebook must be clean")
        if not all(marker in source for marker in REQUIRED_MARKERS):
            raise ValueError("Source notebook is missing full-data markers")
        if any(token in source for token in ("feature_product_ids", "sampled_image_features", "cohort_size", "stable_customer_ids")):
            raise ValueError("Source notebook contains sampled analysis")
        return {"status": "PASS", "mode": mode, "cell_count": len(notebook.cells)}
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if any(cell.execution_count is None for cell in code):
        raise ValueError("Notebook contains unexecuted code cells")
    outputs = [output for cell in code for output in cell.get("outputs", [])]
    if any(output.output_type == "error" for output in outputs):
        raise ValueError("Notebook contains error outputs")
    charts = sum("image/png" in output.get("data", {}) for output in outputs if output.output_type in ("display_data", "execute_result"))
    rendered = "\n".join(str(output.get("text", "")) + str(output.get("data", {})) for output in outputs)
    if any(marker not in rendered for marker in REQUIRED_MARKERS) or "Final execution summary" not in rendered:
        raise ValueError("Executed notebook is missing required evidence")
    if charts < 6:
        raise ValueError("Notebook must contain at least six chart outputs")
    return {"status": "PASS", "mode": mode, "cell_count": len(notebook.cells), "code_cell_count": len(code), "output_count": len(outputs), "chart_count": charts, "error_count": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--mode", choices=("source", "executed"), default="executed")
    args = parser.parse_args()
    summary = inspect_notebook(args.notebook, args.mode)
    summary["notebook_sha256"] = hashlib.sha256(args.notebook.read_bytes()).hexdigest()
    print("\n".join(f"{key}: {value}" for key, value in summary.items()))


if __name__ == "__main__":
    main()
