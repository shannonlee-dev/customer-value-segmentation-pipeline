"""Reusable artifact discovery and validation for the pipeline facade."""

import json
from pathlib import Path

import pandas as pd

from src.runtime import RuntimeContext


class ArtifactStore:
    def __init__(
        self,
        context: RuntimeContext,
        artifact_status: dict[str, str],
        cache_messages: list[str],
    ) -> None:
        self.context = context
        self.artifact_status = artifact_status
        self.cache_messages = cache_messages

    def _reuse_sources(self) -> list[tuple[str, Path]]:
        sources = [("runtime cache", self.context.runtime_root)]
        if self.context.precomputed_root is not None:
            sources.append(("precomputed artifacts", self.context.precomputed_root))
        return sources

    def find_reusable_csv(
        self,
        artifact: str,
        runtime_path: Path,
        filename: str,
        required_columns: tuple[str, ...],
        forbidden_columns: tuple[str, ...] = (),
    ) -> Path | None:
        for source_name, root in self._reuse_sources():
            candidates = (
                [runtime_path]
                if source_name == "runtime cache"
                else sorted(root.rglob(filename))
            )
            for candidate in candidates:
                valid, reason = self._valid_csv(
                    candidate,
                    required_columns,
                    forbidden_columns,
                )
                if valid:
                    self.record_status(artifact, "REUSED", candidate)
                    return candidate
                if candidate.exists():
                    self.record_rejection(artifact, candidate, reason)
        return None

    def find_reusable_json(
        self,
        artifact: str,
        runtime_path: Path,
        filename: str,
    ) -> Path | None:
        for source_name, root in self._reuse_sources():
            candidates = (
                [runtime_path]
                if source_name == "runtime cache"
                else sorted(root.rglob(filename))
            )
            for candidate in candidates:
                try:
                    if candidate.is_file() and isinstance(
                        json.loads(candidate.read_text(encoding="utf-8")),
                        dict,
                    ):
                        self.record_status(artifact, "REUSED", candidate)
                        return candidate
                    if candidate.exists():
                        self.record_rejection(
                            artifact,
                            candidate,
                            "invalid JSON object",
                        )
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                    self.record_rejection(
                        artifact,
                        candidate,
                        f"unreadable JSON ({error})",
                    )
        return None

    @staticmethod
    def _valid_csv(
        path: Path,
        required_columns: tuple[str, ...],
        forbidden_columns: tuple[str, ...] = (),
    ) -> tuple[bool, str]:
        if not path.is_file():
            return False, "file does not exist"
        try:
            sample = pd.read_csv(path, nrows=1)
        except (OSError, UnicodeDecodeError, pd.errors.EmptyDataError, ValueError) as error:
            return False, f"unreadable CSV ({error})"
        missing = set(required_columns).difference(sample.columns)
        if missing:
            return False, f"missing required columns: {sorted(missing)}"
        forbidden = set(forbidden_columns).intersection(sample.columns)
        if forbidden:
            return False, f"contains removed columns: {sorted(forbidden)}"
        if sample.empty:
            return False, "CSV has no data rows"
        return True, ""

    def record_status(self, artifact: str, status: str, path: Path) -> None:
        self.artifact_status[artifact] = status
        self.cache_messages.append(f"{artifact}: {status} ({path})")

    def record_rejection(self, artifact: str, path: Path, reason: str) -> None:
        message = f"{artifact}: REJECTED {path} ({reason})"
        self.cache_messages.append(message)
        print(message)

    @staticmethod
    def csv_row_count(path: Path) -> int:
        with path.open(encoding="utf-8") as source:
            return sum(1 for _ in source) - 1
