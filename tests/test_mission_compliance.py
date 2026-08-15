"""Final public-documentation and repository-boundary audit."""

import re
import shlex
import subprocess
import unittest
from pathlib import Path


README_PATH = Path("README.md")
REQUIRED_TERMS = [
    "H&M Personalized Fashion Recommendations",
    "데이터를 찾은 방법",
    "선택한 이유",
    "라이선스 및 사용 조건",
    "500",
    "SHA-256",
    "NumPy",
    "IQR",
    "그룹별",
    "RFM",
    "비즈니스 인사이트",
    "추가로 필요한 데이터",
    "반증",
    "A/B",
    "연속 메모리",
    "병목",
    "이탈",
]


def forbidden_tracked_paths(paths):
    """Return tracked paths that would expose local data or legacy artifacts."""
    forbidden_basenames = {
        "kaggle.json",
        "ecommerce_transactions.csv",
        "generate_sample_data.py",
    }
    return [
        path
        for path in paths
        if path.startswith(("data/raw/", "data/processed/"))
        or Path(path).suffix.lower() in (".jpg", ".jpeg")
        or Path(path).name.lower() in forbidden_basenames
    ]


def documented_bash_commands(markdown):
    """Parse logical commands from executable Bash fences, excluding prose."""
    bash_blocks = re.findall(r"```bash\n(.*?)```", markdown, flags=re.DOTALL)
    commands = []
    for block in bash_blocks:
        logical_command = ""
        for line in block.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            logical_command += (" " if logical_command else "") + stripped.rstrip("\\").strip()
            if not stripped.endswith("\\"):
                commands.append(shlex.split(logical_command))
                logical_command = ""
        if logical_command:
            commands.append(shlex.split(logical_command))
    return commands


def command_stage(tokens):
    """Classify a documented command by the observable workflow action it performs."""
    command = [token for token in tokens if "=" not in token or token.startswith("--")]
    if command[:4] == ["python3", "-m", "venv", ".venv"]:
        return "create-environment"
    if command[:2] == [".venv/bin/pip", "install"] and "requirements.txt" in command:
        return "install-dependencies"
    if "scripts/prepare_hm_data.py" in command:
        return "prepare-cohort"
    if "scripts/build_notebook.py" in command:
        return "build-notebook"
    if ".venv/bin/jupyter" in command and "nbconvert" in command:
        return "execute-notebook"
    if "scripts/verify_notebook.py" in command:
        return "verify-notebook"
    if "unittest" in command and "discover" in command:
        return "run-tests"
    return None


class MissionComplianceTests(unittest.TestCase):
    """The public guide must preserve required provenance and privacy boundaries."""

    def test_forbidden_tracked_path_helper_catches_nested_and_mixed_case_variants(self):
        """Nested credentials, legacy data, and JPEG variants must not evade the audit."""
        tracked_fixture = [
            "README.md",
            "data/raw/local.csv",
            "data/processed/cohort.csv",
            "assets/lookbook.JPEG",
            "assets/detail.JpG",
            "private/kaggle.json",
            "legacy/ecommerce_transactions.csv",
            "tools/generate_sample_data.py",
            "docs/kaggle.json.example",
        ]

        self.assertEqual(
            forbidden_tracked_paths(tracked_fixture),
            [
                "data/raw/local.csv",
                "data/processed/cohort.csv",
                "assets/lookbook.JPEG",
                "assets/detail.JpG",
                "private/kaggle.json",
                "legacy/ecommerce_transactions.csv",
                "tools/generate_sample_data.py",
            ],
        )

    def test_readme_covers_required_methods_and_boundaries(self):
        """Removing a required workflow concept leaves the public guide incomplete."""
        self.assertTrue(README_PATH.is_file(), "The rebuilt root README does not exist")
        readme = README_PATH.read_text(encoding="utf-8")
        for term in REQUIRED_TERMS:
            self.assertIn(term, readme, "README is missing required term: {!r}".format(term))

    def test_readme_documents_a_runnable_workflow_in_dependency_order(self):
        """The public workflow must build, execute, then verify from a fresh environment."""
        readme = README_PATH.read_text(encoding="utf-8")
        commands = documented_bash_commands(readme)
        stages = [stage for stage in map(command_stage, commands) if stage]
        self.assertEqual(
            stages,
            [
                "create-environment",
                "install-dependencies",
                "prepare-cohort",
                "build-notebook",
                "execute-notebook",
                "verify-notebook",
                "run-tests",
            ],
        )

        by_stage = {
            command_stage(command): command
            for command in commands
            if command_stage(command)
        }
        prepare = by_stage["prepare-cohort"]
        execute = by_stage["execute-notebook"]
        self.assertEqual(prepare[prepare.index("--raw-dir") + 1], "data/raw/h-and-m")
        self.assertIn("HM_RAW_DATA_DIR=data/raw/h-and-m", execute)
        self.assertIn("MPLCONFIGDIR=/tmp/hm-matplotlib-cache", execute)
        self.assertIn("--execute", execute)
        self.assertIn("--inplace", execute)

    def test_tracked_files_exclude_local_data_images_and_credentials(self):
        """Tracking local source artifacts or credentials violates redistribution boundaries."""
        tracked = subprocess.run(
            ["git", "ls-files"], check=True, capture_output=True, text=True
        ).stdout.splitlines()
        self.assertEqual(forbidden_tracked_paths(tracked), [])
