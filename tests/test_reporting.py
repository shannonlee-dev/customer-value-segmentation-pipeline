"""Behavioral tests for full-data statistics and RFM insight reporting."""

import unittest

import numpy as np
import pandas as pd


class ReportingTests(unittest.TestCase):
    def reporting_functions(self):
        try:
            from src.reporting import build_business_insights, summarize_numeric, summarize_rfm_segments
        except ModuleNotFoundError:
            self.fail("src.reporting must provide the report calculations")
        return summarize_numeric, summarize_rfm_segments, build_business_insights

    def test_numeric_summary_uses_the_complete_input_distribution(self):
        summarize_numeric, _, _ = self.reporting_functions()

        result = summarize_numeric(np.array([1.0, 2.0, 3.0, 4.0]))

        self.assertEqual(result["count"], 4)
        self.assertAlmostEqual(result["mean"], 2.5)
        self.assertAlmostEqual(result["median"], 2.5)
        self.assertAlmostEqual(result["std"], 1.2909944487358056)
        self.assertAlmostEqual(result["q1"], 1.75)
        self.assertAlmostEqual(result["q3"], 3.25)

    def test_rfm_summary_and_insights_use_observed_segment_metrics(self):
        _, summarize_rfm_segments, build_business_insights = self.reporting_functions()
        rfm = pd.DataFrame(
            [
                ["c1", 1, 10, 100.0, "VIP"],
                ["c2", 5, 5, 50.0, "Loyal"],
                ["c3", 100, 1, 10.0, "Churned"],
                ["c4", 30, 2, 20.0, "Potential"],
                ["c5", 2, 1, 20.0, "New"],
            ],
            columns=["customer_id", "recency", "frequency", "monetary", "segment"],
        )

        summary = summarize_rfm_segments(rfm)
        insights = build_business_insights(summary)

        self.assertAlmostEqual(summary.loc["VIP", "customer_share_pct"], 20.0)
        self.assertAlmostEqual(summary.loc["VIP", "monetary_share_pct"], 50.0)
        self.assertIn("VIP 고객은 전체 고객의 20.00%", insights)
        self.assertIn("전체 Monetary의 50.00%", insights)
        self.assertEqual(insights.count("**근거**"), 3)
        self.assertEqual(insights.count("**실행**"), 3)
        self.assertEqual(insights.count("**검증**"), 3)


if __name__ == "__main__":
    unittest.main()
