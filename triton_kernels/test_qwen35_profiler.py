import types
import unittest

from triton_kernels.qwen35_profiler import extract_top_ops


class Qwen35ProfilerTests(unittest.TestCase):
    def test_extract_top_ops_sorts_and_computes_share(self):
        events = [
            types.SimpleNamespace(
                key="op_a",
                count=3,
                self_cuda_time_total=80.0,
                cuda_time_total=100.0,
                self_cpu_time_total=10.0,
                cpu_time_total=20.0,
            ),
            types.SimpleNamespace(
                key="op_b",
                count=1,
                self_cuda_time_total=20.0,
                cuda_time_total=30.0,
                self_cpu_time_total=5.0,
                cpu_time_total=8.0,
            ),
        ]

        rows = extract_top_ops(events, sort_key="self_cuda_time_total", row_limit=2)

        self.assertEqual(rows[0]["name"], "op_a")
        self.assertEqual(rows[1]["name"], "op_b")
        self.assertEqual(rows[0]["share_pct"], 80.0)
        self.assertEqual(rows[1]["share_pct"], 20.0)
        self.assertEqual(rows[0]["count"], 3)

    def test_extract_top_ops_handles_missing_metric(self):
        events = [types.SimpleNamespace(key="op", count=1)]

        rows = extract_top_ops(events, sort_key="self_cuda_time_total", row_limit=1)

        self.assertEqual(rows[0]["self_cuda_time_total"], 0.0)
        self.assertEqual(rows[0]["share_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
