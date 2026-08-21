"""Small report calculations shared by the portable notebook and tests."""

import pandas as pd

from .statistics import summarize_numeric


def summarize_rfm_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    """Aggregate observed customer and monetary shares for every RFM segment."""
    required = {"customer_id", "recency", "frequency", "monetary", "segment"}
    missing = required.difference(rfm.columns)
    if missing:
        raise ValueError(f"RFM frame is missing columns: {sorted(missing)}")
    summary = rfm.groupby("segment").agg(
        customers=("customer_id", "size"),
        mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_monetary=("monetary", "mean"),
        monetary_total=("monetary", "sum"),
    )
    summary["customer_share_pct"] = summary["customers"] / summary["customers"].sum() * 100
    monetary_total = summary["monetary_total"].sum()
    summary["monetary_share_pct"] = 0.0 if monetary_total == 0 else summary["monetary_total"] / monetary_total * 100
    return summary.sort_values("customers", ascending=False)


def build_business_insights(segment_summary: pd.DataFrame) -> str:
    """Render three evidence/action/validation insights from observed RFM metrics."""
    def metric(segment: str, column: str) -> float:
        return float(segment_summary.loc[segment, column]) if segment in segment_summary.index else 0.0

    return f"""### 인사이트 1 — VIP 유지

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
- **검증**: 혜택 노출·재고·반품·마진 데이터가 필요하며, 순증 Monetary가 혜택 비용을 넘지 못하면 전략을 반증한다."""
