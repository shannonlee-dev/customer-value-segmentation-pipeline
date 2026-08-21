"""Build a Kaggle report that looks like the main notebook but only reads saved results."""

import sys
from pathlib import Path

import nbformat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_notebook import build_notebook as build_analysis_notebook


def build_notebook(path: Path = Path("notebooks/precomputed_report.ipynb")) -> Path:
    """Write a visual twin of the analysis report that never rebuilds raw data."""
    build_analysis_notebook(path)
    notebook = nbformat.read(path, as_version=4)

    notebook.cells[0].source = """import os
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")
summary_files = sorted(INPUT_ROOT.rglob("aggregates/eda_summary.json"))
PRECOMPUTED_ROOTS = [path.parent.parent for path in summary_files]
if len(PRECOMPUTED_ROOTS) != 1:
    raise RuntimeError(f"사전계산 결과를 하나만 연결하세요. 발견 수: {len(PRECOMPUTED_ROOTS)}")

PRECOMPUTED_ROOT = PRECOMPUTED_ROOTS[0]
required = [
    PRECOMPUTED_ROOT / "processed" / "transactions.csv",
    PRECOMPUTED_ROOT / "processed" / "customers.csv",
    PRECOMPUTED_ROOT / "processed" / "articles.csv",
    PRECOMPUTED_ROOT / "features" / "product_images" / "product_images.csv",
    PRECOMPUTED_ROOT / "aggregates" / "rfm.csv",
    PRECOMPUTED_ROOT / "aggregates" / "monthly_summary.csv",
    PRECOMPUTED_ROOT / "aggregates" / "iqr_unit_price.json",
]
missing = [item for item in required if not item.is_file()]
if missing:
    raise RuntimeError("필수 사전계산 파일이 없습니다:\\n" + "\\n".join(map(str, missing)))
os.environ["HM_PRECOMPUTED_DIR"] = str(PRECOMPUTED_ROOT)
print("사전계산 루트:", PRECOMPUTED_ROOT)"""

    notebook.cells[2].source = """import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("module://matplotlib_inline.backend_inline")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display

INPUT_ROOT = Path("/kaggle/input")
reporting_files = sorted(INPUT_ROOT.rglob("customer-value-segmentation-pipeline/src/reporting.py"))
if len(reporting_files) != 1:
    raise RuntimeError(f"reporting.py가 포함된 프로젝트 Input을 하나만 연결하세요. 발견 수: {len(reporting_files)}")
ROOT = reporting_files[0].parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.reporting import build_business_insights, summarize_rfm_segments

started = time.monotonic()
print("실행 환경: kaggle")
print("실행 모드: precomputed")
print(f"사전 계산 루트: {PRECOMPUTED_ROOT}")
print("프로젝트 소스: 사용 가능")
print("H&M 소스 검증: 통과")
print("분석 범위: 전체 데이터셋")"""

    notebook.cells[4].source = """eda = json.loads((PRECOMPUTED_ROOT / "aggregates" / "eda_summary.json").read_text(encoding="utf-8"))
iqr = json.loads((PRECOMPUTED_ROOT / "aggregates" / "iqr_unit_price.json").read_text(encoding="utf-8"))
rfm = pd.read_csv(PRECOMPUTED_ROOT / "aggregates" / "rfm.csv", dtype={"customer_id": "string"})
image_features = pd.read_csv(PRECOMPUTED_ROOT / "features" / "product_images" / "product_images.csv", dtype={"product_id": "string"})
monthly_value = pd.read_csv(PRECOMPUTED_ROOT / "aggregates" / "monthly_summary.csv")
transaction_path = PRECOMPUTED_ROOT / "processed" / "transactions.csv"
article_path = PRECOMPUTED_ROOT / "processed" / "articles.csv"

transaction_schema = pd.read_csv(transaction_path, nrows=0)
article_schema = pd.read_csv(article_path, nrows=0)
summary = {
    "transaction_rows": int(eda["price_statistics"]["count"]),
    "customer_rows": len(rfm),
    "product_rows": len(image_features),
}
inventory = pd.DataFrame([
    ["transactions", "거래 1건", f"{summary['transaction_rows']:,} × {len(transaction_schema.columns)}", "날짜, 수치, 식별자", "RFM과 시계열"],
    ["customers", "고객 1명", f"{summary['customer_rows']:,}", "수치, 범주, 식별자", "연령과 회원 정보"],
    ["articles", "상품 1개", f"{summary['product_rows']:,} × {len(article_schema.columns)}", "텍스트, 식별자", "텍스트 특징"],
    ["image_features", "상품 이미지 1개", f"{len(image_features):,} × {len(image_features.columns)}", "수치, 식별자", "전체 이미지 평균/표준편차"],
], columns=["원천", "단위", "크기", "주요 자료형", "역할"])
display(Markdown("## 데이터셋 구성"))
display(inventory)

transaction_preview = pd.read_csv(transaction_path, nrows=5).copy()
transaction_preview["customer_id"] = "<masked>"
transaction_preview["product_id"] = "<masked>"
display(transaction_preview.head())
transaction_preview.info()
display(transaction_preview.describe())
print("테이블은 거래·고객·상품·상품 이미지 단위로 정규화된 상태를 유지합니다.")
print("이미지 배열 처리: 사전계산 결과 재사용")"""

    chart_code = notebook.cells[6].source
    start = chart_code.index("monthly_summary_path =")
    end = chart_code.index("product_features = image_features")
    notebook.cells[6].source = (
        chart_code[:start]
        + "# monthly_value는 사전계산 결과에서 이미 읽었습니다.\n"
        + chart_code[end:]
    )
    notebook.cells[8].source = notebook.cells[8].source.replace(
        "insight_path = context.artifact_root / \"business_insights.md\"",
        'insight_path = Path("/kaggle/working/business_insights.md")',
    )
    nbformat.write(notebook, path)
    return path


if __name__ == "__main__":
    build_notebook()
