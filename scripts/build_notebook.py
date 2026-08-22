"""Build the clean, portable, one-click H&M analysis notebook."""

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


PROJECT_BOOTSTRAP_SOURCE = """import os
import subprocess
from pathlib import Path

REPOSITORY_URL = "https://github.com/shannonlee-dev/customer-value-segmentation-pipeline.git"
KAGGLE_PROJECT_ROOT = Path("/kaggle/working/customer-value-segmentation-pipeline")

def is_project_root(path):
    return (path / "src" / "pipeline.py").is_file()

def find_project_root():
    candidates = [Path.cwd(), *Path.cwd().parents]
    if os.environ.get("PROJECT_ROOT"):
        candidates.append(Path(os.environ["PROJECT_ROOT"]))
    for candidate in candidates:
        if is_project_root(candidate):
            return candidate.resolve()

    if not Path("/kaggle").is_dir():
        raise RuntimeError(
            "프로젝트 소스를 찾지 못했습니다. 로컬에서는 자동으로 저장소를 clone하지 않습니다. "
            "PROJECT_ROOT를 설정하거나 저장소 루트에서 노트북을 실행하세요."
        )

    if KAGGLE_PROJECT_ROOT.exists():
        raise RuntimeError(
            f"Kaggle 프로젝트 경로가 이미 있지만 소스가 완전하지 않습니다: {KAGGLE_PROJECT_ROOT}. "
            "해당 폴더를 정리한 뒤 다시 실행하세요."
        )

    print("Kaggle 환경: 프로젝트 소스를 내려받습니다.")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", REPOSITORY_URL, str(KAGGLE_PROJECT_ROOT)],
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "Kaggle에서 프로젝트를 내려받지 못했습니다. Notebook Settings에서 Internet을 켠 뒤 다시 실행하세요."
        ) from error

    if not is_project_root(KAGGLE_PROJECT_ROOT):
        raise RuntimeError(f"clone 뒤에도 프로젝트 소스를 찾지 못했습니다: {KAGGLE_PROJECT_ROOT}")
    return KAGGLE_PROJECT_ROOT.resolve()

PROJECT_ROOT = find_project_root()
os.environ.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))
"""

CHART_STYLE_SOURCE = """KOREAN_CHART_TEXT = {
    "price_title": "상대 가격 분포", "price_x": "상대 데이터셋 가격값", "count_y": "거래 건수",
    "iqr_title": "IQR 필터링 전후 상대 가격", "iqr_x": "IQR 처리 상태", "segment_title": "RFM 세그먼트별 고객 수",
    "segment_x": "RFM 세그먼트", "customer_y": "고객 수", "feature_title": "상품 이미지와 텍스트 특징 상관관계",
    "feature_axis": "특징", "frequency_title": "고객 구매 빈도와 Monetary 값", "frequency_x": "고유 구매일 수",
    "monetary_y": "합계 상대 데이터셋 값", "monthly_title": "월별 합계 상대 데이터셋 값", "month_x": "주문 월",
}
ENGLISH_CHART_TEXT = {
    "price_title": "Relative Price Distribution", "price_x": "Relative dataset price value", "count_y": "Transaction count",
    "iqr_title": "Relative Price Before vs After IQR Filtering", "iqr_x": "IQR treatment state", "segment_title": "RFM Segment Customer Counts",
    "segment_x": "RFM segment", "customer_y": "Customer count", "feature_title": "Product Image and Text Feature Correlation",
    "feature_axis": "Feature", "frequency_title": "Customer Purchase Frequency vs Monetary Value", "frequency_x": "Unique purchase dates",
    "monetary_y": "Aggregate relative dataset value", "monthly_title": "Monthly Aggregate Relative Dataset Value", "month_x": "Order month",
}

required_glyphs = {
    ord(character)
    for label in KOREAN_CHART_TEXT.values()
    for character in label
    if not character.isascii()
}
preferred_families = (
    "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "Malgun Gothic",
    "Apple SD Gothic Neo", "AppleGothic", "Noto Sans CJK JP",
)
font_entries = sorted(
    matplotlib.font_manager.fontManager.ttflist,
    key=lambda font: (
        preferred_families.index(font.name) if font.name in preferred_families else len(preferred_families),
        font.name,
    ),
)

def supports_chart_glyphs(font):
    try:
        face = matplotlib.ft2font.FT2Font(font.fname)
        return all(face.get_char_index(codepoint) for codepoint in required_glyphs)
    except (OSError, RuntimeError):
        return False

korean_font = next((font.name for font in font_entries if supports_chart_glyphs(font)), None)
if korean_font:
    matplotlib.rcParams["font.family"] = korean_font
    matplotlib.rcParams["axes.unicode_minus"] = False
    CHART_TEXT = KOREAN_CHART_TEXT
else:
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    CHART_TEXT = ENGLISH_CHART_TEXT
"""


def build_notebook(path: Path = Path("notebooks/analysis_report.ipynb")) -> Path:
    cells = [
        new_code_cell(PROJECT_BOOTSTRAP_SOURCE + """
precomputed = os.environ.get("HM_PRECOMPUTED_DIR")
if precomputed:
    print("사전 계산 루트:", Path(precomputed).expanduser().resolve())
else:
    print("사전 계산 산출물 없이 실행합니다. 원본 데이터가 있으면 전체 데이터를 계산합니다.")
print("프로젝트 루트:", PROJECT_ROOT)"""),
        new_markdown_cell("""# H&M 고객 가치 분석

이 노트북을 열고 **모두 실행**을 선택하세요. 쓰기 가능한 실행 캐시, `HM_PRECOMPUTED_DIR`, 연결한 Kaggle 전체 데이터 산출물, Kaggle competition 입력, `data/raw/h-and-m`, `HM_RAW_DATA_DIR`을 자동으로 찾습니다. 모든 지표는 사용 가능한 전체 데이터셋을 사용하며 고객·거래·상품·이미지를 표본 추출하지 않습니다.

대용량 원시 데이터 준비와 이미지 특징 추출은 동일한 `DataAnalyzer`가 수행하며, 검증된 **전체 데이터 산출물**을 재사용할 수 있습니다. 산출물이 없으면 같은 코드가 전체 H&M 원본 데이터를 처리합니다."""),
        new_code_cell("""import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("module://matplotlib_inline.backend_inline")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display

""" + CHART_STYLE_SOURCE + """
ROOT = Path(os.environ["PROJECT_ROOT"]).resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.pipeline import DataAnalyzer
from src.reporting import build_business_insights, summarize_numeric, summarize_rfm_segments
from src.runtime import discover_runtime

context = discover_runtime(ROOT)
analyzer = DataAnalyzer(context)
started = time.monotonic()
print(f"실행 환경: {context.runtime_name}")
print(f"실행 모드: {context.runtime_mode}")
if context.precomputed_root is not None:
    print(f"사전 계산 루트: {context.precomputed_root}")
print("프로젝트 소스: 사용 가능")
print("H&M 소스 검증: 통과")
print(f"실행 산출물 루트: {context.runtime_root}")
print("분석 범위: 전체 데이터셋")"""),
        new_markdown_cell("""## 전체 데이터 준비와 멀티모달 특징

거래는 Pandas chunk로 읽습니다. `transactions`, `customers`, `articles`, `image_features`는 각각의 자연스러운 grain을 유지합니다. 이렇게 하면 수천만 거래 행에 고객·상품 속성을 반복하지 않으며, 통계에 필요한 컬럼만 필요한 시점에 chunk 단위로 join합니다.

고객 연령은 고객 grain에서 한 번만 대치합니다. 중앙값은 왜곡된 분포에서 평균보다 덜 민감한 대표값이며, `club_member_status`는 재현 가능한 고객 그룹을 제공합니다. 그룹에 알려진 연령이 없으면 전체 중앙값을 fallback으로 사용합니다. 이 방법은 고객을 보존하지만 분산을 줄이고 그룹 차이를 과장할 수 있습니다.

각 이미지 파일은 `matplotlib.image.imread`로 읽습니다. 파일 loop는 I/O만 수행하고, 이미지 내부 계산은 decode된 전체 배열에 `np.mean`, `np.std`를 적용하는 NumPy vectorized 연산입니다. 이미지를 하나의 전역 tensor로 쌓지 않습니다."""),
        new_code_cell("""summary = analyzer.load_data()
image_features = analyzer.engineer_features()
iqr = analyzer.detect_outliers()
recompute_artifacts = context.raw_data_root is not None
rfm = analyzer.calculate_rfm(force=recompute_artifacts)
eda = analyzer.prepare_eda_artifacts(force=recompute_artifacts)
print(analyzer.format_cache_report())

transaction_schema = pd.read_csv(analyzer.transactions_path, nrows=0)
customers = pd.read_csv(analyzer.customers_path, dtype={"customer_id": "string"})
articles = pd.read_csv(analyzer.articles_path, dtype={"product_id": "string"})
product_features = image_features.copy()
if "product_name_length" not in product_features.columns:
    name_lengths = articles[["product_id", "product_name"]].copy()
    name_lengths["product_name_length"] = name_lengths["product_name"].fillna("").str.len()
    product_features = product_features.merge(
        name_lengths[["product_id", "product_name_length"]],
        on="product_id",
        how="left",
    )
product_features_path = context.feature_root / "product_images_enriched.csv"
product_features.to_csv(product_features_path, index=False)
print(f"이미지·텍스트 특징 저장: {product_features_path}")
inventory = pd.DataFrame([
    ["transactions", "거래 1건", f"{summary['transaction_rows']:,} × {len(transaction_schema.columns)}", "날짜, 수치, 식별자", "RFM과 시계열"],
    ["customers", "고객 1명", f"{len(customers):,} × {len(customers.columns)}", "수치, 범주, 식별자", "연령과 회원 정보"],
    ["articles", "상품 1개", f"{len(articles):,} × {len(articles.columns)}", "텍스트, 식별자", "텍스트 특징"],
    ["image_features", "상품 이미지 1개", f"{len(image_features):,} × {len(image_features.columns)}", "수치, 식별자", "전체 이미지 평균/표준편차"],
], columns=["원천", "단위", "크기", "주요 자료형", "역할"])
display(Markdown("## 데이터셋 구성"))
display(inventory)

transaction_preview = pd.read_csv(analyzer.transactions_path, nrows=5).copy()
transaction_preview["customer_id"] = "<masked>"
transaction_preview["product_id"] = "<masked>"
display(transaction_preview.head())
transaction_preview.info()
display(transaction_preview.describe())
print("테이블은 거래·고객·상품·상품 이미지 단위로 정규화된 상태를 유지합니다.")
print("이미지 배열 처리: matplotlib imread + 전체 배열 NumPy np.mean/np.std")"""),
        new_markdown_cell("""## IQR, 상관관계, 시각 분석

IQR은 전체 가격 분포를 사용합니다. 중간 50%를 사용하므로 극단값에 강건하지만, 자연스럽게 오른쪽으로 긴 가격 분포에서도 정상적인 premium 상품을 이상치로 표시할 수 있습니다. 따라서 아래 전후 boxplot은 진단용이며 데이터를 삭제하라는 지시가 아닙니다.

상관관계 A는 unit_price와 대치된 고객 연령의 거래 가중 상관관계입니다. 상관관계 B는 unit_price와 상품 이미지 평균 밝기의 거래 가중 상관관계입니다. 상관관계는 선형 연관성을 나타낼 뿐 인과관계를 뜻하지 않습니다."""),
        new_code_cell("""price_statistics = eda["price_statistics"]
display(pd.DataFrame([price_statistics], index=["unit_price (전체 거래)"]).round(6))
distribution_note = "평균이 중앙값보다 높아 오른쪽 꼬리 가능성을 보여준다" if price_statistics["mean"] > price_statistics["median"] else "평균과 중앙값의 관계상 강한 오른쪽 꼬리 근거는 제한적이다"
display(Markdown(
    f"### 수치 근거와 해석\\n\\n**기술통계:** 전체 `{price_statistics['count']:,}`건에서 평균은 `{price_statistics['mean']:.6f}`, 중앙값은 `{price_statistics['median']:.6f}`, "
    f"표준편차는 `{price_statistics['std']:.6f}`이며 Q1–Q3는 `{price_statistics['q1']:.6f}–{price_statistics['q3']:.6f}`이다. "
    f"{distribution_note}. 따라서 평균만 보지 않고 중앙값과 사분위 범위를 함께 사용해야 한다."
))
price_age_corr = eda["price_age_correlation"]
monthly_summary_path = eda.get("monthly_summary_path")
if monthly_summary_path is None:
    if context.precomputed_root is None:
        raise KeyError("monthly_summary_path")
    matches = list(context.precomputed_root.rglob("monthly_summary.csv"))
    if len(matches) != 1:
        raise RuntimeError(f"monthly_summary.csv를 하나만 찾아야 합니다. 발견 수: {len(matches)}")
    monthly_summary_path = matches[0]
monthly_value = pd.read_csv(monthly_summary_path)
def transaction_feature_correlation(transaction_path, features, feature_column, lower_fence, upper_fence):
    lookup = features[["product_id", feature_column]].dropna().drop_duplicates("product_id")
    count = sum_x = sum_y = sum_xy = sum_x2 = sum_y2 = 0.0
    outlier_values = set()
    for chunk in pd.read_csv(
        transaction_path,
        usecols=["product_id", "unit_price"],
        dtype={"product_id": "string"},
        chunksize=500_000,
    ):
        prices = chunk["unit_price"]
        outlier_values.update(prices[(prices < lower_fence) | (prices > upper_fence)].unique())
        paired = chunk.merge(lookup, on="product_id", how="inner")
        x = paired["unit_price"].to_numpy(dtype=float)
        y = paired[feature_column].to_numpy(dtype=float)
        count += len(x); sum_x += x.sum(); sum_y += y.sum(); sum_xy += (x * y).sum(); sum_x2 += (x * x).sum(); sum_y2 += (y * y).sum()
    denominator = ((count * sum_x2 - sum_x ** 2) * (count * sum_y2 - sum_y ** 2)) ** 0.5
    correlation = 0.0 if denominator == 0 else float((count * sum_xy - sum_x * sum_y) / denominator)
    return correlation, sorted(outlier_values)

price_image_mean_corr, unique_price_outliers = transaction_feature_correlation(
    analyzer.transactions_path,
    product_features,
    "image_mean",
    iqr["lower_fence"],
    iqr["upper_fence"],
)
price_age_note = "매우 약한 선형 관계" if abs(price_age_corr) < 0.1 else "약한 선형 관계"
price_image_note = "매우 약한 선형 관계" if abs(price_image_mean_corr) < 0.1 else "약한 선형 관계"
display(Markdown(f"**상관관계 1 — 가격·대치 연령:** 거래 가중 상관계수는 `r = {price_age_corr:+.3f}`로 {price_age_note}입니다. 연령만으로 가격을 설명하기 어렵고, 인과관계를 뜻하지 않습니다."))
display(Markdown(f"**상관관계 2 — 가격·이미지 평균:** 거래 가중 상관계수는 `r = {price_image_mean_corr:+.3f}`로 {price_image_note}입니다. 이미지 밝기만으로 가격을 설명하기 어렵고, 인과관계를 뜻하지 않습니다."))

plt.figure(figsize=(8, 4))
plt.stairs(eda["histogram_counts"], eda["histogram_edges"])
plt.title(CHART_TEXT["price_title"])
plt.xlabel(CHART_TEXT["price_x"])
plt.ylabel(CHART_TEXT["count_y"])
plt.show()
display(Markdown(f"**차트 인사이트:** 평균 `{price_statistics['mean']:.6f}`이 중앙값 `{price_statistics['median']:.6f}`보다 {'높아' if price_statistics['mean'] > price_statistics['median'] else '낮아'} 분포의 비대칭성을 확인할 수 있습니다."))

plt.figure(figsize=(8, 4))
ax = plt.gca()
boxplots = [
    {**eda["boxplot_before"], "fliers": unique_price_outliers},
    {**eda["boxplot_after"], "fliers": []},
]
ax.bxp(
    boxplots,
    showfliers=True,
    flierprops={"marker": "o", "markersize": 2, "markerfacecolor": "#d62728", "markeredgecolor": "none", "alpha": 0.35},
)
plt.title(CHART_TEXT["iqr_title"])
plt.xlabel(CHART_TEXT["iqr_x"])
plt.ylabel(CHART_TEXT["price_x"])
plt.show()
display(Markdown(f"**차트 인사이트:** IQR 기준 이상치 `{iqr['outlier_count']:,}`건을 빨간 점으로 표시했으며, 필터링 후 분포는 해당 기준 안의 거래만 포함합니다."))

plt.figure(figsize=(8, 4))
segment_counts = rfm["segment"].value_counts().sort_index()
segment_counts.plot.bar()
plt.title(CHART_TEXT["segment_title"])
plt.xlabel(CHART_TEXT["segment_x"])
plt.ylabel(CHART_TEXT["customer_y"])
plt.show()
largest_segment = segment_counts.idxmax()
display(Markdown(f"**차트 인사이트:** `{largest_segment}` 세그먼트가 `{segment_counts[largest_segment]:,}`명으로 가장 커 고객 구성의 중심을 이룹니다."))

plt.figure(figsize=(6, 5))
feature_correlation = product_features[["image_mean", "image_std", "product_name_length"]].corr()
sns.heatmap(feature_correlation, annot=True, cmap="Blues")
plt.title(CHART_TEXT["feature_title"])
plt.xlabel(CHART_TEXT["feature_axis"])
plt.ylabel(CHART_TEXT["feature_axis"])
plt.show()
feature_pairs = {
    "이미지 평균·이미지 표준편차": feature_correlation.loc["image_mean", "image_std"],
    "이미지 평균·상품명 길이": feature_correlation.loc["image_mean", "product_name_length"],
    "이미지 표준편차·상품명 길이": feature_correlation.loc["image_std", "product_name_length"],
}
strongest_pair, strongest_value = max(feature_pairs.items(), key=lambda item: abs(item[1]))
display(Markdown(f"**차트 인사이트:** 비교한 특징 중 `{strongest_pair}`의 상관계수 절댓값이 `{abs(strongest_value):.3f}`로 가장 큽니다."))

plt.figure(figsize=(8, 4))
plt.scatter(rfm["frequency"], rfm["monetary"], alpha=0.15, s=3, rasterized=True)
plt.title(CHART_TEXT["frequency_title"])
plt.xlabel(CHART_TEXT["frequency_x"])
plt.ylabel(CHART_TEXT["monetary_y"])
plt.show()
frequency_monetary_corr = rfm[["frequency", "monetary"]].corr().iloc[0, 1]
display(Markdown(f"**차트 인사이트:** 구매 빈도와 Monetary의 상관계수는 `r = {frequency_monetary_corr:+.3f}`로, 빈도만으로 고객 가치를 완전히 설명할 수는 없습니다."))

plt.figure(figsize=(9, 4))
plt.plot(monthly_value["order_month"], monthly_value["monetary"])
plt.title(CHART_TEXT["monthly_title"])
plt.xlabel(CHART_TEXT["month_x"])
plt.ylabel(CHART_TEXT["monetary_y"])
plt.xticks(rotation=45)
plt.show()
peak_month = monthly_value.loc[monthly_value["monetary"].idxmax()]
display(Markdown(f"**차트 인사이트:** `{peak_month['order_month']}`의 월별 Monetary가 `{peak_month['monetary']:.2f}`로 가장 높아, 해당 시점의 거래 집중도를 보여줍니다."))"""),
        new_markdown_cell("""## RFM 세분화

분석 기준일은 parameter입니다. 기본값은 마지막 거래일 다음 날이므로 가장 최근 구매자의 Recency는 양수 1일입니다. Frequency는 고유 구매일 수여서 같은 날 여러 상품을 구매한 행이 구매 빈도를 부풀리지 않습니다. Monetary는 상대 거래값의 합계입니다.

R, F, M은 percentile rank로 변환한 후 네 개의 고정 percentile 구간에 배정합니다. 이 방식은 통화별 임의 기준 없이 비교 가능한 1–4점을 만듭니다. Recency는 작을수록, Frequency와 Monetary는 클수록 높은 점수를 받습니다. 동점은 평균 percentile rank를 사용하므로 같은 raw 값의 고객은 항상 같은 점수를 받으며, 점수 그룹별 고객 수가 정확히 같지 않을 수 있습니다.

규칙은 우선순위대로 적용합니다. R = 4, F = 4, M = 4 고객은 **VIP**, R >= 3와 F >= 3 고객은 **Loyal**, 최근 구매한 저빈도 고객은 **New**, R = 1 고객은 **Churned**, 나머지는 **Potential**입니다. business 타당성은 라벨만 보지 않고 관측된 고객 비중, Monetary 비중, 평균 R/F/M으로 평가합니다."""),
        new_code_cell("""segment_summary = summarize_rfm_segments(rfm)
display(segment_summary.round(4))
business_insights = build_business_insights(segment_summary)
display(Markdown("## 실행 시점 비즈니스 인사이트"))
display(Markdown(business_insights))
insight_path = context.artifact_root / "business_insights.md"
insight_path.write_text(business_insights, encoding="utf-8")
print("복사 가능한 비즈니스 인사이트:", insight_path)
display(Markdown("### 최종 실행 요약"))
print("처리한 거래 수:", summary["transaction_rows"])
print("처리한 고객 수:", summary["customer_rows"])
print("처리한 상품 수:", summary["product_rows"])
print("IQR 이상치 수:", iqr["outlier_count"])
print("총 실행 시간(초):", round(time.monotonic() - started, 1))"""),
    ]
    cells.insert(5, new_code_cell("""image_path = analyzer.images_path
article_path = analyzer.articles_path
output_path = Path("/kaggle/working/product_images_enriched.csv")

images = pd.read_csv(image_path, dtype={"product_id": "string"})
articles = pd.read_csv(article_path, dtype={"product_id": "string"})

if "product_name_length" not in articles.columns:
    articles["product_name_length"] = articles["product_name"].fillna("").str.len()
if "product_name_length" in images.columns:
    images = images.drop(columns=["product_name_length"])

product_images_enriched = images.merge(
    articles[["product_id", "product_name_length"]],
    on="product_id",
    how="left",
)

product_images_enriched.to_csv(output_path, index=False)

print(f"저장 완료: {output_path}")
print(product_images_enriched.columns.tolist())
product_images_enriched.head()

image_features = product_images_enriched
product_features = product_images_enriched"""))
    notebook = new_notebook(cells=cells)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, path)
    return path


if __name__ == "__main__":
    build_notebook()
