"""Contract tests for the executed, privacy-safe H&M analysis notebook."""

import re
import tempfile
import unittest
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook, new_output


NOTEBOOK_PATH = Path("notebooks/analysis_report.ipynb")
CUSTOMER_ID_PATTERN = re.compile(r"\b[0-9a-f]{64}\b")
PRODUCT_ID_PATTERN = re.compile(r"\b0\d{9}\b")


def textual_outputs(notebook):
    """Return only renderable text, excluding source and encoded images."""
    values = []
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream":
                values.append(output.get("text", ""))
            elif output.get("output_type") in ("display_data", "execute_result"):
                data = output.get("data", {})
                values.extend(
                    data.get(key, "")
                    for key in ("text/plain", "text/html", "text/markdown")
                )
            elif output.get("output_type") == "error":
                values.extend(output.get("traceback", []))
    return "\n".join(str(value) for value in values)


class ExecutedNotebookContractTests(unittest.TestCase):
    """The committed report remains executed, complete, and redacted."""

    def setUp(self):
        self.assertTrue(
            NOTEBOOK_PATH.is_file(),
            "The rebuilt executed notebook does not exist",
        )
        self.notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)

    def test_notebook_has_the_required_order_and_analysis_operations(self):
        """Removing or reordering an analysis stage breaks the report contract."""
        self.assertEqual(len(self.notebook.cells), 11)
        sources = [cell.source for cell in self.notebook.cells]
        stage_tokens = [
            ("H&M", "non-commercial"),
            ("DataAnalyzer", "HM_RAW_DATA_DIR"),
            (".head(", "safe_preview"),
            (".info(", ".describe("),
            ("missing", "handle_missing_values"),
            ("engineer_features", "product_name_length"),
            ("IQR", "boxplot"),
            ("correlation", "describe"),
            ("histplot", "countplot", "heatmap", "scatterplot", "lineplot"),
            ("calculate_rfm", "segment"),
            ("Markdown", "limitations"),
        ]
        for index, tokens in enumerate(stage_tokens):
            for token in tokens:
                self.assertIn(token, sources[index], "cell {} must contain {!r}".format(index + 1, token))

        all_source = "\n".join(sources)
        required_tokens = [
            "DataAnalyzer", ".head(", ".info(", ".describe(",
            "histplot", "boxplot", "countplot", "heatmap",
            "scatterplot", "lineplot", "calculate_rfm",
        ]
        for token in required_tokens:
            self.assertIn(token, all_source)

    def test_notebook_contains_required_explanations(self):
        """Removing a required interpretation leaves the report incomplete."""
        all_source = "\n".join(cell.source for cell in self.notebook.cells)
        for token in ("H&M", "IQR", "그룹별", "NumPy", "RFM", "표본", "price"):
            self.assertIn(token, all_source)
        self.assertIn("relative", all_source.lower())

    def test_every_code_cell_is_executed_without_errors_and_six_charts_exist(self):
        """An unexecuted stage, exception, or missing chart invalidates the artifact."""
        code_cells = [cell for cell in self.notebook.cells if cell.cell_type == "code"]
        self.assertTrue(code_cells)
        self.assertTrue(
            all(type(cell.execution_count) is int for cell in code_cells),
            "Every code cell must have an integer execution count",
        )
        outputs = [output for cell in code_cells for output in cell.get("outputs", [])]
        self.assertFalse(any(output.get("output_type") == "error" for output in outputs))
        chart_count = sum(
            "image/png" in output.get("data", {})
            for output in outputs
            if output.get("output_type") in ("display_data", "execute_result")
        )
        self.assertGreaterEqual(chart_count, 6)

    def test_textual_outputs_do_not_expose_customer_or_product_identifiers(self):
        """A rendered raw identifier is a redistribution/privacy regression."""
        rendered_text = textual_outputs(self.notebook)
        self.assertIsNone(CUSTOMER_ID_PATTERN.search(rendered_text))
        self.assertIsNone(PRODUCT_ID_PATTERN.search(rendered_text))

    def test_structural_preview_contains_only_type_placeholders(self):
        """Calling head must never render a raw row or raw product name."""
        preview_outputs = textual_outputs(new_notebook(cells=[self.notebook.cells[2]]))
        expected_placeholders = {
            "<date>", "<masked-id>", "<text>", "<category>", "<numeric>",
            "<numeric-or-missing>", "<category-or-missing>", "<local-image-path>",
        }
        for placeholder in expected_placeholders:
            self.assertIn(placeholder, preview_outputs)

    def test_correlation_outputs_interpret_direction_strength_and_practical_magnitude(self):
        """Bare coefficients cannot substitute for two computed interpretations."""
        markdown = "\n".join(
            output.get("data", {}).get("text/markdown", "")
            for output in self.notebook.cells[7].get("outputs", [])
        )
        interpretation_pattern = re.compile(
            r"^- (?P<label>[^:]+): r = (?P<coefficient>[+-]\d+\.\d{3}); "
            r"(?P<strength>negligible|weak|moderate|strong) "
            r"(?P<direction>positive|negative|zero) association\. "
            r"Squared magnitude = (?P<squared>\d+\.\d{2})%\. This indicates "
            r"(?P<signal>little|limited|noticeable|substantial) standalone linear signal",
            re.MULTILINE,
        )
        interpretations = {
            match.group("label"): match.groupdict()
            for match in interpretation_pattern.finditer(markdown)
        }
        self.assertEqual(
            set(interpretations),
            {"Relative price and age", "Image Mean and product-name length"},
        )
        for interpretation in interpretations.values():
            coefficient = float(interpretation["coefficient"])
            squared_percent = float(interpretation["squared"])
            magnitude = abs(coefficient)
            expected_direction = "positive" if coefficient > 0 else "negative" if coefficient < 0 else "zero"
            expected_strength = (
                "negligible" if magnitude < 0.10 else "weak" if magnitude < 0.30
                else "moderate" if magnitude < 0.50 else "strong"
            )
            expected_signal = (
                "little" if squared_percent < 1 else "limited" if squared_percent < 9
                else "noticeable" if squared_percent < 25 else "substantial"
            )
            self.assertEqual(interpretation["direction"], expected_direction)
            self.assertEqual(interpretation["strength"], expected_strength)
            self.assertEqual(interpretation["signal"], expected_signal)
            self.assertAlmostEqual(squared_percent, coefficient ** 2 * 100, delta=0.02)
        self.assertEqual(markdown.count("standalone linear signal"), 2)
        self.assertIn("observational", markdown)

    def test_text_length_covers_full_cohort_while_images_stay_sampled(self):
        """Text length must cover every loaded row even when image decoding is sampled."""
        setup_outputs = textual_outputs(new_notebook(cells=[self.notebook.cells[1]]))
        loaded_match = re.search(r"Loaded aggregate scope: ([\d,]+) transactions", setup_outputs)
        self.assertIsNotNone(loaded_match)
        loaded_rows = int(loaded_match.group(1).replace(",", ""))

        feature_outputs = self.notebook.cells[5].get("outputs", [])
        markdown = "\n".join(
            output.get("data", {}).get("text/markdown", "")
            for output in feature_outputs
        )
        text_coverage = re.search(
            r"Product-name length coverage: ([\d,]+) cohort transactions",
            markdown,
        )
        image_coverage = re.search(
            r"Image Mean/Std coverage: ([\d,]+) sampled products out of ([\d,]+), "
            r"joined back to ([\d,]+) transaction rows",
            markdown,
        )
        self.assertIsNotNone(text_coverage)
        self.assertIsNotNone(image_coverage)
        self.assertEqual(int(text_coverage.group(1).replace(",", "")), loaded_rows)
        self.assertEqual(int(image_coverage.group(1).replace(",", "")), 64)
        self.assertGreater(int(image_coverage.group(2).replace(",", "")), 64)
        self.assertEqual(int(image_coverage.group(3).replace(",", "")), 81)
        self.assertIn("transaction-weighted", markdown)

        plain_tables = [
            output.get("data", {}).get("text/plain", "")
            for output in feature_outputs
            if output.get("data", {}).get("text/html")
        ]
        self.assertEqual(len(plain_tables), 2)
        text_count_match = re.search(r"count\s+([\d.]+)", plain_tables[1])
        self.assertIsNotNone(text_count_match)
        self.assertEqual(int(float(text_count_match.group(1))), loaded_rows)

    def test_image_interpretations_state_the_transaction_weighted_join_back_scope(self):
        """Image summaries must not read like unweighted 64-product statistics."""
        for cell_index in (7, 10):
            rendered_markdown = "\n".join(
                output.get("data", {}).get("text/markdown", "")
                for output in self.notebook.cells[cell_index].get("outputs", [])
            )
            self.assertIn("64", rendered_markdown)
            self.assertIn("81", rendered_markdown)
            self.assertIn("transaction-weighted", rendered_markdown)


class NotebookVerifierTests(unittest.TestCase):
    """The verifier detects execution and redaction regressions."""

    def setUp(self):
        self.assertTrue(NOTEBOOK_PATH.is_file(), "Notebook required before verifier tests")
        from scripts.verify_notebook import format_evidence, inspect_notebook

        self.format_evidence = format_evidence
        self.inspect_notebook = inspect_notebook

    def _write_notebook(self, notebook):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        path = Path(temporary_directory.name) / "candidate.ipynb"
        nbformat.write(notebook, path)
        return path

    def _valid_minimal_notebook(self):
        outputs = [
            new_output(
                output_type="display_data",
                data={"image/png": "aW1hZ2U=", "text/plain": "<Figure>"},
                metadata={},
            )
            for _ in range(6)
        ]
        return new_notebook(cells=[new_code_cell("pass", execution_count=1, outputs=outputs)])

    def test_inspection_summarizes_the_real_executed_notebook(self):
        """A valid report produces auditable counts and a PASS status."""
        summary = self.inspect_notebook(NOTEBOOK_PATH)
        self.assertEqual(summary["cell_count"], 11)
        self.assertGreaterEqual(summary["chart_count"], 6)
        self.assertEqual(summary["error_count"], 0)
        self.assertEqual(summary["redaction_status"], "PASS")
        self.assertEqual(summary["status"], "PASS")

    def test_inspection_rejects_unexecuted_cells_and_identifier_outputs(self):
        """Missing execution evidence and leaked IDs cannot receive PASS."""
        unexecuted = self._write_notebook(new_notebook(cells=[new_code_cell("pass")]))
        with self.assertRaisesRegex(ValueError, "unexecuted"):
            self.inspect_notebook(unexecuted)

        leaked = self._valid_minimal_notebook()
        leaked.cells[0].outputs.append(
            new_output(output_type="stream", name="stdout", text="0" * 64)
        )
        with self.assertRaisesRegex(ValueError, "identifier"):
            self.inspect_notebook(self._write_notebook(leaked))

    def test_inspection_rejects_identifier_in_static_markdown(self):
        """A raw identifier in authored Markdown must invalidate the notebook."""
        leaked = self._valid_minimal_notebook()
        leaked.cells.insert(0, new_markdown_cell("Customer: {}".format("a" * 64)))

        with self.assertRaisesRegex(ValueError, "identifier"):
            self.inspect_notebook(self._write_notebook(leaked))

    def test_inspection_rejects_identifier_in_any_textual_output_mime(self):
        """Rendered text leaks must be rejected regardless of their text MIME subtype."""
        for mime_type, identifier in (
            ("text/markdown", "0000000001"),
            ("text/csv", "b" * 64),
        ):
            with self.subTest(mime_type=mime_type):
                leaked = self._valid_minimal_notebook()
                leaked.cells[0].outputs.append(
                    new_output(
                        output_type="display_data",
                        data={mime_type: identifier},
                        metadata={},
                    )
                )

                with self.assertRaisesRegex(ValueError, "identifier"):
                    self.inspect_notebook(self._write_notebook(leaked))

    def test_inspection_does_not_treat_png_payload_as_text(self):
        """Identifier-shaped bytes in an encoded PNG are not rendered text."""
        candidate = self._valid_minimal_notebook()
        candidate.cells[0].outputs[0]["data"]["image/png"] = "0" * 64

        summary = self.inspect_notebook(self._write_notebook(candidate))

        self.assertEqual(summary["redaction_status"], "PASS")

    def test_inspection_rejects_errors_and_fewer_than_six_charts(self):
        """Error outputs and incomplete visualization evidence are rejected."""
        errored = self._valid_minimal_notebook()
        errored.cells[0].outputs.append(
            new_output(
                output_type="error",
                ename="RuntimeError",
                evalue="boom",
                traceback=["RuntimeError: boom"],
            )
        )
        with self.assertRaisesRegex(ValueError, "error"):
            self.inspect_notebook(self._write_notebook(errored))

        incomplete = self._valid_minimal_notebook()
        incomplete.cells[0].outputs.pop()
        with self.assertRaisesRegex(ValueError, "six"):
            self.inspect_notebook(self._write_notebook(incomplete))

    def test_evidence_text_is_deterministic_and_contains_the_notebook_digest(self):
        """Repeated formatting yields the same hash-backed audit record."""
        summary = self.inspect_notebook(NOTEBOOK_PATH)
        first = self.format_evidence(NOTEBOOK_PATH, summary)
        second = self.format_evidence(NOTEBOOK_PATH, summary)
        self.assertEqual(first, second)
        self.assertRegex(first, r"notebook_sha256: [0-9a-f]{64}")
        self.assertIn("status: PASS", first)


if __name__ == "__main__":
    unittest.main()
