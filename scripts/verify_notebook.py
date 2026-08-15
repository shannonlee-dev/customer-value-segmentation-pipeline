"""Verify notebook execution, chart evidence, and output redaction."""

import argparse
import hashlib
import re
from pathlib import Path

import nbformat


DEFAULT_NOTEBOOK_PATH = Path("notebooks/analysis_report.ipynb")
DEFAULT_LOG_PATH = Path("artifacts/notebook_execution.log")
CUSTOMER_ID_PATTERN = re.compile(r"\b[0-9a-f]{64}\b")
PRODUCT_ID_PATTERN = re.compile(r"\b0\d{9}\b")


def _textual_outputs(notebook):
    values = []
    for cell in notebook.cells:
        if cell.cell_type == "markdown":
            values.append(cell.get("source", ""))
        for output in cell.get("outputs", []):
            output_type = output.get("output_type")
            if output_type == "stream":
                values.append(output.get("text", ""))
            elif output_type in ("display_data", "execute_result"):
                data = output.get("data", {})
                values.extend(
                    value
                    for mime_type, value in data.items()
                    if mime_type.startswith("text/")
                )
            elif output_type == "error":
                values.extend(output.get("traceback", []))
    return "\n".join(str(value) for value in values)


def inspect_notebook(path):
    """Return execution counts for a valid notebook or reject unsafe evidence."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Notebook does not exist: {}".format(path))

    notebook = nbformat.read(path, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    unexecuted = [
        index
        for index, cell in enumerate(code_cells, start=1)
        if type(cell.execution_count) is not int
    ]
    if unexecuted:
        raise ValueError("Notebook contains unexecuted code cells: {}".format(unexecuted))

    outputs = [output for cell in code_cells for output in cell.get("outputs", [])]
    error_count = sum(output.get("output_type") == "error" for output in outputs)
    if error_count:
        raise ValueError("Notebook contains {} error output(s)".format(error_count))

    chart_count = sum(
        "image/png" in output.get("data", {})
        for output in outputs
        if output.get("output_type") in ("display_data", "execute_result")
    )
    if chart_count < 6:
        raise ValueError("Notebook must contain at least six image/png chart outputs")

    rendered_text = _textual_outputs(notebook)
    if CUSTOMER_ID_PATTERN.search(rendered_text) or PRODUCT_ID_PATTERN.search(rendered_text):
        raise ValueError("Notebook textual outputs contain a raw identifier")

    return {
        "cell_count": len(notebook.cells),
        "code_cell_count": len(code_cells),
        "output_count": len(outputs),
        "chart_count": chart_count,
        "error_count": error_count,
        "unexecuted_cell_count": len(unexecuted),
        "redaction_status": "PASS",
        "status": "PASS",
    }


def format_evidence(path, summary):
    """Format a deterministic SHA-backed verification record."""
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    keys = (
        "cell_count",
        "code_cell_count",
        "output_count",
        "chart_count",
        "error_count",
        "unexecuted_cell_count",
        "redaction_status",
        "status",
    )
    lines = ["notebook: {}".format(path.as_posix()), "notebook_sha256: {}".format(digest)]
    lines.extend("{}: {}".format(key, summary[key]) for key in keys)
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", nargs="?", type=Path, default=DEFAULT_NOTEBOOK_PATH)
    parser.add_argument("--no-write", action="store_true")
    arguments = parser.parse_args()

    summary = inspect_notebook(arguments.notebook)
    evidence = format_evidence(arguments.notebook, summary)
    if not arguments.no_write:
        DEFAULT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_LOG_PATH.write_text(evidence, encoding="utf-8")
    print(evidence, end="")


if __name__ == "__main__":
    main()
