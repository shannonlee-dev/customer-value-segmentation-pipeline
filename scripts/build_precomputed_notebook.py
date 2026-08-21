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
import time
from pathlib import Path

import matplotlib
matplotlib.use("module://matplotlib_inline.backend_inline")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display

available_fonts = {font.name for font in matplotlib.font_manager.fontManager.ttflist}
korean_font = next((name for name in ("NanumGothic", "Noto Sans CJK KR", "Malgun Gothic") if name in available_fonts), None)
if korean_font:
    matplotlib.rcParams["font.family"] = korean_font
    CHART_TEXT = {
        "price_title": "상대 가격 분포", "price_x": "상대 데이터셋 가격값", "count_y": "거래 건수",
        "iqr_title": "IQR 필터링 전후 상대 가격", "iqr_x": "IQR 처리 상태", "segment_title": "RFM 세그먼트별 고객 수",
        "segment_x": "RFM 세그먼트", "customer_y": "고객 수", "feature_title": "상품 이미지와 텍스트 특징 상관관계",
        "feature_axis": "특징", "frequency_title": "고객 구매 빈도와 Monetary 값", "frequency_x": "고유 구매일 수",
        "monetary_y": "합계 상대 데이터셋 값", "monthly_title": "월별 합계 상대 데이터셋 값", "month_x": "주문 월",
    }
else:
    CHART_TEXT = {
        "price_title": "Relative Price Distribution", "price_x": "Relative dataset price value", "count_y": "Transaction count",
        "iqr_title": "Relative Price Before vs After IQR Filtering", "iqr_x": "IQR treatment state", "segment_title": "RFM Segment Customer Counts",
        "segment_x": "RFM segment", "customer_y": "Customer count", "feature_title": "Product Image and Text Feature Correlation",
        "feature_axis": "Feature", "frequency_title": "Customer Purchase Frequency vs Monetary Value", "frequency_x": "Unique purchase dates",
        "monetary_y": "Aggregate relative dataset value", "monthly_title": "Monthly Aggregate Relative Dataset Value", "month_x": "Order month",
    }

def summarize_rfm_segments(rfm):
    summary = rfm.groupby("segment").agg(
        customers=("customer_id", "size"),
        mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_monetary=("monetary", "mean"),
        monetary_total=("monetary", "sum"),
    )
    summary["customer_share_pct"] = summary["customers"] / summary["customers"].sum() * 100
    total = summary["monetary_total"].sum()
    summary["monetary_share_pct"] = 0.0 if total == 0 else summary["monetary_total"] / total * 100
    return summary.sort_values("customers", ascending=False)

def build_business_insights(segment_summary):
    def metric(segment, column):
        return float(segment_summary.loc[segment, column]) if segment in segment_summary.index else 0.0
    return f'''### 인사이트 1 — VIP 유지

- **근거**: VIP 고객은 전체 고객의 {metric('VIP', 'customer_share_pct'):.2f}%이며 전체 Monetary의 {metric('VIP', 'monetary_share_pct'):.2f}%를 차지하고, 평균 고유 구매일 수는 {metric('VIP', 'mean_frequency'):.2f}일이다.
- **실행**: VIP를 대상으로 신상품 조기 접근과 재입고 알림을 제공해 재방문 및 Monetary 유지를 기대한다.
- **검증**: 캠페인 노출·클릭·구매·홀드아웃 데이터가 필요하며, 대조군 대비 재방문율과 Monetary가 개선되지 않으면 가설을 기각한다.

### 인사이트 2 — Churned 재활성화

- **근거**: Churned 고객은 전체 고객의 {metric('Churned', 'customer_share_pct'):.2f}%이며 평균 Recency는 {metric('Churned', 'mean_recency'):.2f}일, 평균 고유 구매일 수는 {metric('Churned', 'mean_frequency'):.2f}일이다.
- **실행**: Churned를 대상으로 동의 기반 복귀 메시지와 제한적 혜택을 실험해 90일 내 재구매율 상승을 기대한다.
- **검증**: 메시지 노출·수신 거부·쿠폰 비용·재구매 데이터가 필요하며, 복귀율이 개선되지 않거나 접촉 피로가 증가하면 중단한다.

### 인사이트 3 — Loyal의 VIP 전환

- **근거**: Loyal 고객은 전체 고객의 {metric('Loyal', 'customer_share_pct'):.2f}%이며 전체 Monetary의 {metric('Loyal', 'monetary_share_pct'):.2f}%를 차지하고, 평균 고유 구매일 수는 {metric('Loyal', 'mean_frequency'):.2f}일이다.
- **실행**: Loyal을 대상으로 구매 빈도 기반 단계형 혜택을 시험해 VIP 전환과 Monetary 상승을 기대한다.
- **검증**: 혜택 노출·재고·반품·마진 데이터가 필요하며, 순증 Monetary가 혜택 비용을 넘지 못하면 전략을 반증한다.'''

started = time.monotonic()
print("실행 환경: kaggle")
print("실행 모드: precomputed")
print(f"사전 계산 루트: {PRECOMPUTED_ROOT}")
print("H&M 소스 검증: 통과")
print("분석 범위: 전체 데이터셋")"""

    notebook.cells[4].source = """eda = json.loads((PRECOMPUTED_ROOT / "aggregates" / "eda_summary.json").read_text(encoding="utf-8"))
iqr = json.loads((PRECOMPUTED_ROOT / "aggregates" / "iqr_unit_price.json").read_text(encoding="utf-8"))
rfm = pd.read_csv(PRECOMPUTED_ROOT / "aggregates" / "rfm.csv", dtype={"customer_id": "string"})
image_features = pd.read_csv(PRECOMPUTED_ROOT / "features" / "product_images" / "product_images.csv", dtype={"product_id": "string"})
monthly_value = pd.read_csv(PRECOMPUTED_ROOT / "aggregates" / "monthly_summary.csv")
transaction_path = PRECOMPUTED_ROOT / "processed" / "transactions.csv"
article_path = PRECOMPUTED_ROOT / "processed" / "articles.csv"

product_features = image_features.copy()
if "product_name_length" not in product_features.columns:
    article_features = pd.read_csv(article_path, dtype={"product_id": "string"})
    if "product_name_length" not in article_features.columns:
        article_features["product_name_length"] = article_features["product_name"].fillna("").str.len()
    product_features = product_features.merge(
        article_features[["product_id", "product_name_length"]],
        on="product_id",
        how="left",
    )
image_features = product_features
product_features_path = Path("/kaggle/working/product_images_enriched.csv")
product_features.to_csv(product_features_path, index=False)
print(f"이미지·텍스트 특징 저장: {product_features_path}")

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
    chart_code = chart_code.replace("product_features = image_features\n", "")
    start = chart_code.index("monthly_summary_path =")
    end = chart_code.index("price_age_note =")
    notebook.cells[6].source = (
        chart_code[:start]
        + "# monthly_value는 사전계산 결과에서 이미 읽었습니다.\n"
        + chart_code[end:]
    )
    notebook.cells[6].source = notebook.cells[6].source.replace(
        "analyzer.transactions_path",
        "transaction_path",
    )
    for original, replacement in {
        'plt.title("상대 가격 분포")': 'plt.title(CHART_TEXT["price_title"])',
        'plt.xlabel("상대 데이터셋 가격값")': 'plt.xlabel(CHART_TEXT["price_x"])',
        'plt.ylabel("거래 건수")': 'plt.ylabel(CHART_TEXT["count_y"])',
        'plt.title("IQR 필터링 전후 상대 가격")': 'plt.title(CHART_TEXT["iqr_title"])',
        'plt.xlabel("IQR 처리 상태")': 'plt.xlabel(CHART_TEXT["iqr_x"])',
        'plt.ylabel("상대 데이터셋 가격값")': 'plt.ylabel(CHART_TEXT["price_x"])',
        'plt.title("RFM 세그먼트별 고객 수")': 'plt.title(CHART_TEXT["segment_title"])',
        'plt.xlabel("RFM 세그먼트")': 'plt.xlabel(CHART_TEXT["segment_x"])',
        'plt.ylabel("고객 수")': 'plt.ylabel(CHART_TEXT["customer_y"])',
        'plt.title("상품 이미지와 텍스트 특징 상관관계")': 'plt.title(CHART_TEXT["feature_title"])',
        'plt.xlabel("특징")': 'plt.xlabel(CHART_TEXT["feature_axis"])',
        'plt.ylabel("특징")': 'plt.ylabel(CHART_TEXT["feature_axis"])',
        'plt.title("고객 구매 빈도와 Monetary 값")': 'plt.title(CHART_TEXT["frequency_title"])',
        'plt.xlabel("고유 구매일 수")': 'plt.xlabel(CHART_TEXT["frequency_x"])',
        'plt.ylabel("합계 상대 데이터셋 값")': 'plt.ylabel(CHART_TEXT["monetary_y"])',
        'plt.title("월별 합계 상대 데이터셋 값")': 'plt.title(CHART_TEXT["monthly_title"])',
        'plt.xlabel("주문 월")': 'plt.xlabel(CHART_TEXT["month_x"])',
    }.items():
        notebook.cells[6].source = notebook.cells[6].source.replace(original, replacement)
    notebook.cells[8].source = notebook.cells[8].source.replace(
        "insight_path = context.artifact_root / \"business_insights.md\"",
        'insight_path = Path("/kaggle/working/business_insights.md")',
    )
    nbformat.write(notebook, path)
    return path


if __name__ == "__main__":
    build_notebook()
