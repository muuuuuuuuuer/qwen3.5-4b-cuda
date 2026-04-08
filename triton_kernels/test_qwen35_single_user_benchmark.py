import unittest

from triton_kernels.qwen35_single_user_benchmark import (
    build_round_robin_mode_orders,
    compare_single_user_results,
    summarize_latency_trace,
    summarize_mode_position_counts,
)


class SingleUserBenchmarkSummaryTests(unittest.TestCase):
    def test_summarize_latency_trace_reports_expected_stats(self):
        summary = summarize_latency_trace([1.0, 2.0, 3.0, 4.0])

        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["min_ms"], 1.0)
        self.assertEqual(summary["max_ms"], 4.0)
        self.assertEqual(summary["mean_ms"], 2.5)
        self.assertEqual(summary["p50_ms"], 2.5)
        self.assertEqual(summary["p90_ms"], 4.0)

    def test_summarize_latency_trace_handles_empty_input(self):
        summary = summarize_latency_trace([])

        self.assertEqual(summary["count"], 0)
        self.assertIsNone(summary["mean_ms"])
        self.assertIsNone(summary["p50_ms"])
        self.assertIsNone(summary["p90_ms"])


class SingleUserBenchmarkCompareTests(unittest.TestCase):
    def test_compare_single_user_results_reports_speedups_and_generation_match(self):
        reference = {
            "ttft_summary": {"mean_ms": 120.0},
            "decode_summary": {"mean_ms": 40.0},
            "end_to_end_summary": {"mean_ms": 680.0},
            "generated_token_ids": [11, 22, 33],
        }
        candidate = {
            "ttft_summary": {"mean_ms": 100.0},
            "decode_summary": {"mean_ms": 32.0},
            "end_to_end_summary": {"mean_ms": 540.0},
            "generated_token_ids": [11, 22, 33],
        }

        comparison = compare_single_user_results(reference, candidate)

        self.assertEqual(comparison["ttft_speedup"], 1.2)
        self.assertEqual(comparison["decode_mean_speedup"], 1.25)
        self.assertEqual(comparison["end_to_end_speedup"], round(680.0 / 540.0, 6))
        self.assertTrue(comparison["same_generation"])

    def test_compare_single_user_results_detects_generation_difference(self):
        reference = {
            "ttft_summary": {"mean_ms": 100.0},
            "decode_summary": {"mean_ms": 30.0},
            "end_to_end_summary": {"mean_ms": 400.0},
            "generated_token_ids": [1, 2, 3],
        }
        candidate = {
            "ttft_summary": {"mean_ms": 100.0},
            "decode_summary": {"mean_ms": 30.0},
            "end_to_end_summary": {"mean_ms": 400.0},
            "generated_token_ids": [1, 2, 4],
        }

        comparison = compare_single_user_results(reference, candidate)

        self.assertFalse(comparison["same_generation"])


class SingleUserBenchmarkSchedulingTests(unittest.TestCase):
    def test_build_round_robin_mode_orders_rotates_mode_priority(self):
        orders = build_round_robin_mode_orders(["fla", "triton_base", "triton_fused"], cycles=5)

        self.assertEqual(
            orders,
            [
                ["fla", "triton_base", "triton_fused"],
                ["triton_base", "triton_fused", "fla"],
                ["triton_fused", "fla", "triton_base"],
                ["fla", "triton_base", "triton_fused"],
                ["triton_base", "triton_fused", "fla"],
            ],
        )

    def test_summarize_mode_position_counts_balances_full_cycle(self):
        orders = build_round_robin_mode_orders(["fla", "triton_base", "triton_fused"], cycles=3)

        summary = summarize_mode_position_counts(orders)

        self.assertEqual(summary["fla"], {0: 1, 1: 1, 2: 1})
        self.assertEqual(summary["triton_base"], {0: 1, 1: 1, 2: 1})
        self.assertEqual(summary["triton_fused"], {0: 1, 1: 1, 2: 1})


if __name__ == "__main__":
    unittest.main()
