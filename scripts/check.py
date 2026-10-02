"""외부 서비스에 접속하지 않고 문법과 로컬 문서 링크를 검사한다."""

import ast
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {
    ".git",
    ".venv",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".cache",
}


def main():
    errors = []
    count = 0
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or EXCLUDED.intersection(path.relative_to(ROOT).parts):
            continue
        try:
            if path.suffix == ".py":
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                count += 1
            elif path.suffix == ".json":
                json.loads(path.read_text(encoding="utf-8"))
                count += 1
            elif path.suffix == ".sh":
                result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
                if result.returncode:
                    errors.append(result.stderr.strip())
                count += 1
            elif path.suffix == ".md":
                # 보고서 템플릿의 상대 링크는 렌더링된 출력 경로를 기준으로 합니다.
                if "src" in path.relative_to(ROOT).parts and "templates" in path.parts:
                    continue
                text = path.read_text(encoding="utf-8")
                text = re.sub(r"(?ms)^(`{3,}|~{3,})[^\n]*\n.*?^\1[^\n]*$", "", text)
                for target in re.findall(r"!?\[[^\]\n]*\]\(([^\n]+?)\)", text):
                    target = target.strip()
                    if target.startswith("<"):
                        target = target[1 : target.index(">")]
                    else:
                        target = target.split(' "', 1)[0]
                    parsed = urlsplit(target)
                    if parsed.scheme or parsed.netloc or not parsed.path:
                        continue
                    destination = (
                        ROOT / unquote(parsed.path.lstrip("/"))
                        if parsed.path.startswith("/")
                        else path.parent / unquote(parsed.path)
                    )
                    if destination.is_file() and destination.suffix == ".md" and parsed.fragment:
                        content = destination.read_text(encoding="utf-8")
                        content = re.sub(r"(?ms)^(`{3,}|~{3,})[^\n]*\n.*?^\1[^\n]*$", "", content)
                        anchors = set()
                        for heading in re.findall(r"^#{1,6}\s+(.+?)\s*$", content, re.MULTILINE):
                            cleaned = re.sub(r"[^\w\-\s]", "", heading.lower().replace("`", ""))
                            anchors.add(re.sub(r"\s", "-", cleaned))
                        if unquote(parsed.fragment) not in anchors:
                            errors.append(f"{path.relative_to(ROOT)}: 없는 문서 앵커 {target}")
                    if not destination.exists():
                        errors.append(f"{path.relative_to(ROOT)}: 없는 링크 대상 {target}")
                count += 1
        except (SyntaxError, ValueError, OSError) as exc:
            errors.append(f"{path.relative_to(ROOT)}: {exc}")
    headings = re.findall(r"^#{1,2} .+$", (ROOT / "README.md").read_text(encoding="utf-8"), re.M)
    if len(headings) < 4 or headings[1:4] != [
        "## 프로젝트 소개",
        "## 핵심 특징",
        "## 아키텍처",
    ]:
        errors.append("README 첫 섹션 순서: 프로젝트명 → 프로젝트 소개 → 핵심 특징 → 아키텍처")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    section = readme.split("## 아키텍처", 1)[-1].split("\n## ", 1)[0]
    if not re.search(r"(?m)^\|.+\|$", section):
        errors.append("README 아키텍처에는 구성 요소와 역할을 설명하는 표가 필요합니다.")
    if not re.search(r"(?m)^\|.+\|\n\s*\n```mermaid\n", section):
        errors.append("README 아키텍처 표 바로 아래에 Mermaid 다이어그램이 필요합니다.")
    for error in errors:
        print(error, file=sys.stderr)
    print(f"문법·문서 검사: {count}개 파일, 오류 {len(errors)}개")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
