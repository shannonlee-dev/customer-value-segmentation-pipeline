"""Verify notebook execution, chart evidence, and output redaction."""

import argparse
import hashlib
from pathlib import Path

import nbformat

if __package__:
    from .constants import (
        CHART_MIME_TYPE,
        CODE_CELL_TYPE,
        CUSTOMER_ID_PATTERN,
        DEFAULT_LOG_PATH,
        DEFAULT_NOTEBOOK_PATH,
        EVIDENCE_SUMMARY_KEYS,
        MINIMUM_CHART_COUNT,
        PASS_STATUS,
        PRODUCT_ID_PATTERN,
        RENDERED_OUTPUT_TYPES,
        TEXT_MIME_PREFIX,
    )
else:
    from constants import (
        CHART_MIME_TYPE,
        CODE_CELL_TYPE,
        CUSTOMER_ID_PATTERN,
        DEFAULT_LOG_PATH,
        DEFAULT_NOTEBOOK_PATH,
        EVIDENCE_SUMMARY_KEYS,
        MINIMUM_CHART_COUNT,
        PASS_STATUS,
        PRODUCT_ID_PATTERN,
        RENDERED_OUTPUT_TYPES,
        TEXT_MIME_PREFIX,
    )


def _textual_outputs(notebook):
    values = []
    for cell in notebook.cells:
        if cell.cell_type == "markdown":
            values.append(cell.get("source", ""))
        for output in cell.get("outputs", []):
            output_type = output.get("output_type")
            if output_type == "stream":
                values.append(output.get("text", ""))
            elif output_type in RENDERED_OUTPUT_TYPES:
                data = output.get("data", {})
                values.extend(
                    value
                    for mime_type, value in data.items()
                    if mime_type.startswith(TEXT_MIME_PREFIX)
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
    code_cells = [cell for cell in notebook.cells if cell.cell_type == CODE_CELL_TYPE]
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
        CHART_MIME_TYPE in output.get("data", {})
        for output in outputs
        if output.get("output_type") in RENDERED_OUTPUT_TYPES
    )
    if chart_count < MINIMUM_CHART_COUNT:
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
        "redaction_status": PASS_STATUS,
        "status": PASS_STATUS,
    }


def format_evidence(path, summary):
    """Format a deterministic SHA-backed verification record."""
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    lines = ["notebook: {}".format(path.as_posix()), "notebook_sha256: {}".format(digest)]
    lines.extend("{}: {}".format(key, summary[key]) for key in EVIDENCE_SUMMARY_KEYS)
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
