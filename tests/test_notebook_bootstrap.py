"""Contract for Kaggle Input project-source discovery."""

import json
import unittest
from pathlib import Path


class NotebookBootstrapTests(unittest.TestCase):
    def test_notebook_builder_declares_kaggle_input_bootstrap(self):
        source = Path("scripts/build_notebook.py").read_text(encoding="utf-8")
        self.assertIn('BASE = Path("/kaggle/input/notebooks/classichit/codyssey-a-1-1")', source)
        self.assertIn('BASE.rglob("src/pipeline.py")', source)
        self.assertIn('os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)', source)

    def test_committed_notebook_starts_with_kaggle_input_bootstrap(self):
        notebook = json.loads(Path("notebooks/analysis_report.ipynb").read_text(encoding="utf-8"))
        first_cell = notebook["cells"][0]
        source = "".join(first_cell["source"])
        self.assertEqual(first_cell["cell_type"], "code")
        self.assertIn('BASE = Path("/kaggle/input/notebooks/classichit/codyssey-a-1-1")', source)
        self.assertIn('BASE.rglob("src/pipeline.py")', source)
        self.assertIn('os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)', source)


if __name__ == "__main__":
    unittest.main()
