"""Create a tiny H&M-shaped fixture and verify notebook Run All without real data."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import image as mpimg

from build_notebook import build_notebook


def write_fixture(raw: Path) -> None:
    image_dir = raw / "images" / "001"
    image_dir.mkdir(parents=True)
    pd.DataFrame([
        ["2020-01-01", "customer-a", "0010000001", 0.10, 1], ["2020-01-02", "customer-a", "0010000002", 0.20, 1],
        ["2020-01-03", "customer-b", "0010000001", 0.30, 2], ["2020-01-04", "customer-c", "0010000003", 0.50, 2],
    ], columns=["t_dat", "customer_id", "article_id", "price", "sales_channel_id"]).to_csv(raw / "transactions_train.csv", index=False)
    pd.DataFrame([
        ["0010000001", "Item one", "Group A"], ["0010000002", "Item two", "Group B"], ["0010000003", "No image", "Group C"],
    ], columns=["article_id", "prod_name", "product_group_name"]).to_csv(raw / "articles.csv", index=False)
    pd.DataFrame([
        ["customer-a", 25, "ACTIVE", "Regularly"], ["customer-b", None, "ACTIVE", "None"], ["customer-c", 45, "PRE-CREATE", "Monthly"],
    ], columns=["customer_id", "age", "club_member_status", "fashion_news_frequency"]).to_csv(raw / "customers.csv", index=False)
    mpimg.imsave(image_dir / "0010000001.jpg", np.full((10, 8, 3), 0.2))
    mpimg.imsave(image_dir / "0010000002.jpg", np.full((6, 7), 0.7), cmap="gray")


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        raw, runtime = root / "raw", root / "runtime"
        raw.mkdir()
        write_fixture(raw)
        notebook = build_notebook(root / "analysis_report.ipynb")
        environment = os.environ | {
            "PROJECT_ROOT": str(project), "HM_RAW_DATA_DIR": str(raw), "HM_RUNTIME_DIR": str(runtime),
            "MPLCONFIGDIR": str(root / "mpl"), "JUPYTER_CONFIG_DIR": str(root / "jupyter-config"),
            "JUPYTER_DATA_DIR": str(root / "jupyter-data"), "JUPYTER_RUNTIME_DIR": str(root / "jupyter-runtime"),
        }
        subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", str(notebook), "--ExecutePreprocessor.timeout=120"], check=True, cwd=project, env=environment)
        subprocess.run([sys.executable, "scripts/verify_notebook.py", str(notebook), "--mode", "executed"], check=True, cwd=project, env=environment)


if __name__ == "__main__":
    main()
