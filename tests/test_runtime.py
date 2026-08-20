"""Runtime discovery contract tests for the portable full-data pipeline."""

import tempfile
import unittest
from pathlib import Path


class RuntimeDiscoveryTests(unittest.TestCase):
    def _source(self, root):
        for name in ("transactions_train.csv", "customers.csv", "articles.csv"):
            (root / name).write_text("header\n", encoding="utf-8")
        (root / "images").mkdir()

    def test_environment_source_and_runtime_override_take_priority(self):
        from src.runtime import discover_runtime

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            explicit = root / "explicit"
            explicit.mkdir()
            self._source(explicit)
            runtime = root / "runtime"
            context = discover_runtime(
                project_root=root,
                environ={"HM_RAW_DATA_DIR": str(explicit), "HM_RUNTIME_DIR": str(runtime)},
                kaggle_root=root / "missing-kaggle",
            )
            self.assertEqual(context.runtime_name, "local")
            self.assertEqual(context.raw_data_root, explicit)
            self.assertEqual(context.runtime_root, runtime)
            self.assertTrue(context.processed_root.is_dir())

    def test_kaggle_candidate_precedes_project_local_default(self):
        from src.runtime import discover_runtime

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            kaggle_root = root / "kaggle"
            candidate = kaggle_root / "input" / "competitions" / "h-and-m-personalized-fashion-recommendations"
            candidate.mkdir(parents=True)
            self._source(candidate)
            local = root / "data" / "raw" / "h-and-m"
            local.mkdir(parents=True)
            self._source(local)
            context = discover_runtime(project_root=root, environ={}, kaggle_root=kaggle_root)

        self.assertEqual(context.runtime_name, "kaggle")
        self.assertEqual(context.raw_data_root, candidate)
        self.assertEqual(context.runtime_root, kaggle_root / "working" / "hm-customer-value")

    def test_precomputed_environment_takes_priority_over_default_root(self):
        from src.runtime import discover_runtime

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw = root / "raw"
            raw.mkdir()
            self._source(raw)
            precomputed = root / "precomputed"
            precomputed.mkdir()

            context = discover_runtime(
                project_root=root,
                environ={"HM_RAW_DATA_DIR": str(raw), "HM_PRECOMPUTED_DIR": str(precomputed)},
                kaggle_root=root / "missing-kaggle",
            )

        self.assertEqual(context.precomputed_root, precomputed)
        self.assertEqual(context.runtime_mode, "precomputed")

    def test_precomputed_root_is_not_used_as_runtime_output(self):
        from src.runtime import discover_runtime

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw = root / "raw"
            raw.mkdir()
            self._source(raw)
            precomputed = root / "precomputed"
            precomputed.mkdir()

            context = discover_runtime(
                project_root=root,
                environ={"HM_RAW_DATA_DIR": str(raw), "HM_PRECOMPUTED_DIR": str(precomputed)},
                kaggle_root=root / "missing-kaggle",
            )

            self.assertNotEqual(context.runtime_root, precomputed)
            self.assertTrue(context.runtime_root.is_dir())

    def test_missing_dataset_has_actionable_error(self):
        from src.runtime import discover_runtime

        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "HM_RAW_DATA_DIR"):
                discover_runtime(
                    project_root=Path(temporary_directory),
                    environ={},
                    kaggle_root=Path(temporary_directory) / "missing-kaggle",
                )


if __name__ == "__main__":
    unittest.main()
