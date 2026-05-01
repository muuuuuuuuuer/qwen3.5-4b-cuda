import tempfile
import unittest
import json
from pathlib import Path

import benchmark_qwen35_gsm8k_final as gsm8k


class GSM8KFinalBenchmarkHelpersTest(unittest.TestCase):
    def test_extract_answer_prefers_final_answer_phrase(self):
        text = "We compute 12 + 30 = 42. The answer is $1,234.5"

        self.assertEqual(gsm8k.extract_answer(text), "1234.5")

    def test_extract_answer_falls_back_to_last_number(self):
        text = "First try gives 18, but correcting the count gives -7."

        self.assertEqual(gsm8k.extract_answer(text), "-7")

    def test_normalize_gt_reads_gsm8k_marker(self):
        self.assertEqual(gsm8k.normalize_gt("some rationale\n#### 1,008"), "1008")

    def test_is_correct_compares_numeric_strings(self):
        self.assertTrue(gsm8k.is_correct("42.0", "42"))
        self.assertFalse(gsm8k.is_correct(None, "42"))

    def test_build_prompt_inserts_question_at_answer_slot(self):
        prompt = gsm8k.build_prompt("How many apples remain?")

        self.assertIn("Question: How many apples remain?\nAnswer:", prompt)
        self.assertTrue(prompt.rstrip().endswith("Answer:"))

    def test_summarize_results_uses_latency_and_token_counts(self):
        rows = [
            {
                "correct": True,
                "predicted": "1",
                "prefill_ms": 10.0,
                "decode_mean_ms_per_token": 2.0,
                "total_ms": 30.0,
                "generated_tokens": 11,
            },
            {
                "correct": False,
                "predicted": None,
                "prefill_ms": 20.0,
                "decode_mean_ms_per_token": 4.0,
                "total_ms": 60.0,
                "generated_tokens": 21,
            },
        ]

        summary = gsm8k.summarize_results(rows)

        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["invalid_count"], 1)
        self.assertEqual(summary["prefill_mean_ms"], 15.0)
        self.assertEqual(summary["decode_median_ms_per_token"], 3.0)
        self.assertEqual(summary["generated_tokens_total"], 32)

    def test_question_id_cache_is_reused(self):
        dataset = [{"question": str(i), "answer": f"#### {i}"} for i in range(10)]
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "ids.json"

            first = gsm8k.prepare_question_rows(dataset, num_questions=5, seed=42, cache_path=cache_path)
            second = gsm8k.prepare_question_rows(dataset, num_questions=5, seed=7, cache_path=cache_path)

        self.assertEqual([row["question_id"] for row in first], [row["question_id"] for row in second])

    def test_eager_packed_mode_is_not_static_or_compiled(self):
        spec = gsm8k.resolve_mode_spec(gsm8k.EAGER_PACKED_MODE)

        self.assertEqual(spec["deltanet_mode"], "triton_lowrank_beta_gate_packed")
        self.assertFalse(spec["use_static_cache"])
        self.assertFalse(spec["compile_decode"])

    def test_eval_modes_only_keep_operator_path_to_packed(self):
        self.assertEqual(
            gsm8k.EVAL_MODES,
            [
                "torch",
                "fla",
                "fp16_eager",
                gsm8k.EAGER_PACKED_MODE,
                gsm8k.STATIC_COMPILED_MODE,
                gsm8k.STATIC_COMPILED_PACKED_MODE,
            ],
        )

        static_compiled = gsm8k.resolve_mode_spec(gsm8k.STATIC_COMPILED_MODE)
        static_compiled_packed = gsm8k.resolve_mode_spec(gsm8k.STATIC_COMPILED_PACKED_MODE)

        self.assertEqual(static_compiled["deltanet_mode"], "triton_fused")
        self.assertTrue(static_compiled["use_static_cache"])
        self.assertTrue(static_compiled["compile_decode"])
        self.assertEqual(static_compiled_packed["deltanet_mode"], "triton_lowrank_beta_gate_packed")
        self.assertTrue(static_compiled_packed["use_static_cache"])
        self.assertTrue(static_compiled_packed["compile_decode"])

        for removed_mode in ("fp16_static_eager_packed_fullattn_qkv_fused",):
            with self.assertRaises(ValueError):
                gsm8k.resolve_mode_spec(removed_mode)

    def test_collect_summary_payloads_merges_existing_json(self):
        existing_payload = {
            "mode": "torch",
            "summary": {"decode_mean_ms_per_token": 10.0},
        }
        updated_payload = {
            "mode": gsm8k.EAGER_PACKED_MODE,
            "summary": {"decode_mean_ms_per_token": 5.0},
        }
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            gsm8k.output_path_for_mode(output_dir, "torch", False).write_text(json.dumps(existing_payload))

            merged = gsm8k.collect_summary_payloads(
                output_dir,
                smoke=False,
                updated_payloads={gsm8k.EAGER_PACKED_MODE: updated_payload},
                summary_modes=("torch", gsm8k.EAGER_PACKED_MODE),
            )

        self.assertEqual(set(merged), {"torch", gsm8k.EAGER_PACKED_MODE})
        self.assertEqual(merged["torch"], existing_payload)
        self.assertEqual(merged[gsm8k.EAGER_PACKED_MODE], updated_payload)


if __name__ == "__main__":
    unittest.main()
