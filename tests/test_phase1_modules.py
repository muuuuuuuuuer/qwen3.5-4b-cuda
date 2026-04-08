import json
import tempfile
import unittest
from pathlib import Path

import torch

from deltanet_diagnostics import (
    capture_forward_io,
    find_first_deltanet_layer,
    list_module_tensors,
    summarize_past_key_values,
)
from cpu_reference import (
    check_correctness,
    quantized_matmul_reference,
    quantized_matvec_reference,
    self_test,
)
from phase1_utils import (
    build_layer_record,
    extract_text_config,
    generate_text_smoke,
    select_phase1_benchmark_rows,
)
from quantize import (
    attach_layer_metadata,
    dequantize_int8,
    quantize_error_analysis,
    quantize_layer_records,
    quantize_symmetric_int8,
    summarize_errors_by_type,
)


class ConfigExtractionTests(unittest.TestCase):
    def test_extract_text_config_reads_nested_text_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "model_type": "qwen3_5",
                        "text_config": {
                            "hidden_size": 2560,
                            "intermediate_size": 9216,
                            "num_hidden_layers": 32,
                            "layer_types": ["linear_attention", "full_attention"],
                        },
                    }
                ),
                encoding="utf-8",
            )

            text_config = extract_text_config(config_path)

            self.assertEqual(text_config["hidden_size"], 2560)
            self.assertEqual(text_config["layer_types"][1], "full_attention")


class LayerClassificationTests(unittest.TestCase):
    def test_build_layer_record_marks_mlp_proj_as_p0(self):
        layer_types = ["linear_attention"] * 32
        module = torch.nn.Linear(2560, 9216, bias=False)

        record = build_layer_record("model.layers.0.mlp.gate_proj", module, layer_types)

        self.assertIsNotNone(record)
        self.assertEqual(record["layer_index"], 0)
        self.assertEqual(record["layer_type"], "linear_attention")
        self.assertEqual(record["priority"], "P0-优先")
        self.assertEqual(record["shape"], "[9216, 2560]")
        self.assertEqual(record["params"], 9216 * 2560)
        self.assertEqual(record["sub_type"], "FFN")
        self.assertEqual(record["proj_name"], "gate_proj")
        self.assertEqual(record["out_features"], 9216)
        self.assertEqual(record["in_features"], 2560)
        self.assertAlmostEqual(record["params_M"], 9216 * 2560 / 1e6)

    def test_build_layer_record_marks_full_attention_proj_as_p1(self):
        layer_types = ["linear_attention"] * 32
        layer_types[3] = "full_attention"
        module = torch.nn.Linear(2560, 4096, bias=False)

        record = build_layer_record("model.layers.3.self_attn.q_proj", module, layer_types)

        self.assertIsNotNone(record)
        self.assertEqual(record["layer_type"], "full_attention")
        self.assertEqual(record["priority"], "P1-次优")
        self.assertEqual(record["sub_type"], "FullAttn")
        self.assertEqual(record["proj_name"], "q_proj")

    def test_build_layer_record_marks_linear_attention_proj_as_p2(self):
        layer_types = ["linear_attention"] * 32
        module = torch.nn.Linear(2560, 4096, bias=False)

        record = build_layer_record("model.layers.0.self_attn.q_proj", module, layer_types)

        self.assertIsNotNone(record)
        self.assertEqual(record["layer_type"], "linear_attention")
        self.assertEqual(record["priority"], "P2-暂缓")
        self.assertEqual(record["sub_type"], "DeltaNet_Attn")
        self.assertEqual(record["proj_name"], "q_proj")

    def test_build_layer_record_marks_deltanet_proj_as_p2(self):
        layer_types = ["linear_attention"] * 32
        module = torch.nn.Linear(2560, 8192, bias=False)

        record = build_layer_record(
            "model.language_model.layers.0.linear_attn.in_proj_qkv",
            module,
            layer_types,
        )

        self.assertIsNotNone(record)
        self.assertEqual(record["layer_type"], "linear_attention")
        self.assertEqual(record["priority"], "P2-暂缓")
        self.assertEqual(record["sub_type"], "DeltaNet_Attn")
        self.assertEqual(record["proj_name"], "in_proj_qkv")

    def test_build_layer_record_skips_visual_layers(self):
        layer_types = ["linear_attention"] * 32
        module = torch.nn.Linear(1024, 1024, bias=False)

        record = build_layer_record("visual.blocks.0.attn.q_proj", module, layer_types)

        self.assertIsNone(record)

    def test_select_phase1_benchmark_rows_picks_expected_targets(self):
        rows = [
            {"submodule_name": "ffn_gate", "sub_type": "FFN", "proj_name": "gate_proj", "shape": "[9216, 2560]"},
            {"submodule_name": "ffn_down", "sub_type": "FFN", "proj_name": "down_proj", "shape": "[2560, 9216]"},
            {"submodule_name": "full_q", "sub_type": "FullAttn", "proj_name": "q_proj", "shape": "[8192, 2560]"},
            {"submodule_name": "full_k", "sub_type": "FullAttn", "proj_name": "k_proj", "shape": "[1024, 2560]"},
            {
                "submodule_name": "delta_qkv",
                "sub_type": "DeltaNet_Attn",
                "proj_name": "in_proj_qkv",
                "shape": "[8192, 2560]",
            },
            {
                "submodule_name": "delta_z",
                "sub_type": "DeltaNet_Attn",
                "proj_name": "in_proj_z",
                "shape": "[4096, 2560]",
            },
        ]

        selected = select_phase1_benchmark_rows(rows)

        self.assertEqual(
            [row["benchmark_name"] for row in selected],
            [
                "FFN_gate_proj",
                "FFN_down_proj",
                "FullAttn_q_proj",
                "FullAttn_k_proj",
                "DeltaNet_in_proj_qkv",
                "DeltaNet_in_proj_z",
            ],
        )
        self.assertEqual(selected[0]["submodule_name"], "ffn_gate")
        self.assertEqual(selected[-1]["submodule_name"], "delta_z")


class QuantizationTests(unittest.TestCase):
    def test_quantize_symmetric_int8_returns_expected_values(self):
        weight = torch.tensor([[1.27, -1.27], [0.0, 0.635]], dtype=torch.float16)

        qweight, scale = quantize_symmetric_int8(weight)

        self.assertTrue(torch.equal(qweight, torch.tensor([[127, -127], [0, 127]], dtype=torch.int8)))
        self.assertTrue(torch.allclose(scale.float(), torch.tensor([0.01, 0.005]), atol=1e-3))

    def test_dequantize_round_trip_stays_close(self):
        weight = torch.tensor([[1.27, -1.27], [0.0, 0.635]], dtype=torch.float16)

        qweight, scale = quantize_symmetric_int8(weight)
        restored = dequantize_int8(qweight, scale)

        self.assertTrue(torch.allclose(restored, weight.float(), atol=2e-3))

    def test_quantize_error_analysis_reports_small_error_for_exact_values(self):
        weight = torch.tensor([[1.27, -1.27], [0.0, 0.635]], dtype=torch.float16)

        qweight, scale = quantize_symmetric_int8(weight)
        errors = quantize_error_analysis(weight, qweight, scale)

        self.assertLess(errors["max_abs_err"], 2e-3)
        self.assertLess(errors["mean_abs_err"], 2e-3)

    def test_quantize_layer_records_preserves_layer_metadata(self):
        class FakeBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = torch.nn.Module()
                self.mlp.gate_proj = torch.nn.Linear(2, 3, bias=False)

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.layers = torch.nn.ModuleList([FakeBlock()])

        model = FakeModel()
        rows = [
            {
                "submodule_name": "model.layers.0.mlp.gate_proj",
                "sub_type": "FFN",
                "proj_name": "gate_proj",
                "priority": "P0-优先",
            }
        ]

        results = quantize_layer_records(model, rows)

        self.assertEqual(list(results.keys()), ["model.layers.0.mlp.gate_proj"])
        result = results["model.layers.0.mlp.gate_proj"]
        self.assertEqual(result["sub_type"], "FFN")
        self.assertEqual(result["proj_name"], "gate_proj")
        self.assertEqual(result["priority"], "P0-优先")
        self.assertEqual(result["shape"], [3, 2])

    def test_summarize_errors_by_type_aggregates_counts_and_means(self):
        results = {
            "layer_a": {
                "sub_type": "FFN",
                "errors": {"max_abs_err": 1.0, "mean_abs_err": 0.5, "mean_rel_err": 0.25},
            },
            "layer_b": {
                "sub_type": "FFN",
                "errors": {"max_abs_err": 3.0, "mean_abs_err": 1.5, "mean_rel_err": 0.75},
            },
            "layer_c": {
                "sub_type": "FullAttn",
                "errors": {"max_abs_err": 2.0, "mean_abs_err": 1.0, "mean_rel_err": 0.5},
            },
        }

        summary = summarize_errors_by_type(results)
        by_type = {row["sub_type"]: row for row in summary}

        self.assertEqual(by_type["FFN"]["count"], 2)
        self.assertAlmostEqual(by_type["FFN"]["avg_max_abs_err"], 2.0)
        self.assertAlmostEqual(by_type["FFN"]["avg_mean_abs_err"], 1.0)
        self.assertAlmostEqual(by_type["FFN"]["avg_mean_rel_err"], 0.5)
        self.assertEqual(by_type["FullAttn"]["count"], 1)

    def test_attach_layer_metadata_backfills_existing_quantized_results(self):
        results = {
            "layer_a": {
                "qweight": torch.ones((2, 2), dtype=torch.int8),
                "scale": torch.ones(2, dtype=torch.float16),
                "errors": {"max_abs_err": 1.0, "mean_abs_err": 0.5, "mean_rel_err": 0.25},
                "shape": [2, 2],
            }
        }
        rows = [
            {
                "submodule_name": "layer_a",
                "sub_type": "FFN",
                "proj_name": "gate_proj",
                "priority": "P0-优先",
                "layer_index": 0,
            }
        ]

        hydrated = attach_layer_metadata(results, rows)

        self.assertEqual(hydrated["layer_a"]["sub_type"], "FFN")
        self.assertEqual(hydrated["layer_a"]["proj_name"], "gate_proj")
        self.assertEqual(hydrated["layer_a"]["priority"], "P0-优先")
        self.assertEqual(hydrated["layer_a"]["layer_index"], 0)


class CpuReferenceTests(unittest.TestCase):
    def test_quantized_matmul_reference_matches_manual_result(self):
        x = torch.tensor([[1.0, 2.0]], dtype=torch.float16)
        qweight = torch.tensor([[10, -10], [20, 5]], dtype=torch.int8)
        scale = torch.tensor([0.1, 0.2], dtype=torch.float16)

        y = quantized_matmul_reference(x, qweight, scale)

        expected = x.float() @ (qweight.float() * scale.float().unsqueeze(1)).T
        self.assertTrue(torch.allclose(y, expected))

    def test_quantized_matvec_reference_matches_single_vector_case(self):
        x = torch.tensor([1.0, 2.0], dtype=torch.float16)
        qweight = torch.tensor([[10, -10], [20, 5]], dtype=torch.int8)
        scale = torch.tensor([0.1, 0.2], dtype=torch.float16)

        y = quantized_matvec_reference(x, qweight, scale)

        expected = (qweight.float() * scale.float().unsqueeze(1)) @ x.float()
        self.assertTrue(torch.allclose(y, expected))

    def test_check_correctness_reports_perfect_match(self):
        y_ref = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
        y_test = y_ref.clone()

        report = check_correctness(y_ref, y_test, tag="unit")

        self.assertAlmostEqual(report["max_abs_err"], 0.0)
        self.assertAlmostEqual(report["mean_abs_err"], 0.0)
        self.assertAlmostEqual(report["rel_err"], 0.0)
        self.assertAlmostEqual(report["cos_sim"], 1.0, places=6)

    def test_self_test_returns_reports_for_each_requested_shape(self):
        reports = self_test(
            test_shapes=[
                ("tiny_ffn", 8, 4),
                ("tiny_full", 6, 4),
                ("tiny_delta", 5, 4),
            ]
        )

        self.assertEqual([report["tag"] for report in reports], ["tiny_ffn", "tiny_full", "tiny_delta"])
        self.assertTrue(all(report["cos_sim"] > 0.99 for report in reports))


class TextGenerationSmokeTests(unittest.TestCase):
    def test_generate_text_smoke_uses_chat_template_and_returns_only_new_tokens(self):
        class FakeTokenizer:
            def __init__(self):
                self.chat_template_calls = []
                self.encoded_texts = []
                self.decode_inputs = []

            def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
                self.chat_template_calls.append(
                    {
                        "messages": messages,
                        "tokenize": tokenize,
                        "add_generation_prompt": add_generation_prompt,
                        "enable_thinking": enable_thinking,
                    }
                )
                return "CHAT_PROMPT"

            def __call__(self, text, return_tensors="pt"):
                self.encoded_texts.append(text)
                if text == "CHAT_PROMPT":
                    return {
                        "input_ids": torch.tensor([[11, 12, 13]], dtype=torch.long),
                        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
                    }
                if text == "Hello, how are you?":
                    return {
                        "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
                        "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
                    }
                raise AssertionError(f"Unexpected prompt: {text}")

            def decode(self, token_ids, skip_special_tokens=True):
                if isinstance(token_ids, torch.Tensor):
                    token_ids = token_ids.tolist()
                self.decode_inputs.append((token_ids, skip_special_tokens))
                return ",".join(str(token_id) for token_id in token_ids)

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.generate_calls = []

            def generate(self, **kwargs):
                self.generate_calls.append(kwargs)
                return torch.tensor([[11, 12, 13, 90, 91]], dtype=torch.long)

        tokenizer = FakeTokenizer()
        model = FakeModel()

        text = generate_text_smoke(model, tokenizer, prompt="Hello, how are you?", max_new_tokens=2)

        self.assertEqual(text, "90,91")
        self.assertEqual(tokenizer.encoded_texts, ["CHAT_PROMPT"])
        self.assertEqual(
            tokenizer.chat_template_calls,
            [
                {
                    "messages": [{"role": "user", "content": "Hello, how are you?"}],
                    "tokenize": False,
                    "add_generation_prompt": True,
                    "enable_thinking": False,
                }
            ],
        )
        self.assertEqual(tokenizer.decode_inputs, [([90, 91], True)])
        self.assertEqual(model.generate_calls[0]["max_new_tokens"], 2)


class DeltaNetDiagnosticsTests(unittest.TestCase):
    def test_find_first_deltanet_layer_prefers_wrapped_model_path(self):
        linear_attn = torch.nn.Linear(4, 4, bias=False)

        class WrappedLanguageModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([torch.nn.Module()])
                self.layers[0].linear_attn = linear_attn

        class WrappedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.language_model = WrappedLanguageModel()

        class RootModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = WrappedModel()

        info = find_first_deltanet_layer(RootModel())

        self.assertEqual(info["attr_path"], "model.language_model.layers[0].linear_attn")
        self.assertEqual(info["module_name"], "model.language_model.layers.0.linear_attn")
        self.assertIs(info["module"], linear_attn)

    def test_find_first_deltanet_layer_falls_back_to_direct_language_model(self):
        linear_attn = torch.nn.Linear(4, 4, bias=False)

        class DirectLanguageModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([torch.nn.Module()])
                self.layers[0].linear_attn = linear_attn

        class RootModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.language_model = DirectLanguageModel()

        info = find_first_deltanet_layer(RootModel())

        self.assertEqual(info["attr_path"], "language_model.layers[0].linear_attn")
        self.assertEqual(info["module_name"], "language_model.layers.0.linear_attn")
        self.assertIs(info["module"], linear_attn)

    def test_list_module_tensors_reports_parameters_and_buffers(self):
        class TinyDeltaNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(3, 5, bias=False)
                self.register_buffer("state", torch.zeros(2, 3, dtype=torch.float16))

        rows = list_module_tensors(TinyDeltaNet())

        self.assertEqual(
            rows,
            [
                {
                    "kind": "parameter",
                    "name": "proj.weight",
                    "shape": [5, 3],
                    "dtype": "torch.float32",
                },
                {
                    "kind": "buffer",
                    "name": "state",
                    "shape": [2, 3],
                    "dtype": "torch.float16",
                },
            ],
        )

    def test_capture_forward_io_records_tensor_shapes(self):
        module = torch.nn.Linear(3, 2, bias=False)
        x = torch.randn(4, 3)

        calls = capture_forward_io(module, lambda: module(x))

        self.assertEqual(len(calls), 1)
        self.assertEqual(
            calls[0],
            {
                "inputs": [
                    {
                        "name": "inputs[0]",
                        "shape": [4, 3],
                        "dtype": "torch.float32",
                    }
                ],
                "outputs": [
                    {
                        "name": "output",
                        "shape": [4, 2],
                        "dtype": "torch.float32",
                    }
                ],
            },
        )

    def test_capture_forward_io_records_tensor_kwargs(self):
        class KwargOnlyModule(torch.nn.Module):
            def forward(self, *, hidden_states):
                return hidden_states + 1

        module = KwargOnlyModule()
        x = torch.randn(2, 3)

        calls = capture_forward_io(module, lambda: module(hidden_states=x))

        self.assertEqual(len(calls), 1)
        self.assertEqual(
            calls[0],
            {
                "inputs": [
                    {
                        "name": "kwargs.hidden_states",
                        "shape": [2, 3],
                        "dtype": "torch.float32",
                    }
                ],
                "outputs": [
                    {
                        "name": "output",
                        "shape": [2, 3],
                        "dtype": "torch.float32",
                    }
                ],
            },
        )

    def test_summarize_past_key_values_handles_tuples_and_mappings(self):
        cache = [
            (
                torch.zeros(1, 2, 3, dtype=torch.float16),
                torch.ones(1, 2, 4, dtype=torch.float16),
            ),
            {"state": torch.zeros(5, 6, dtype=torch.float32)},
        ]

        summary = summarize_past_key_values(cache)

        self.assertEqual(
            summary,
            [
                {
                    "name": "layer 0[0]",
                    "shape": [1, 2, 3],
                    "dtype": "torch.float16",
                },
                {
                    "name": "layer 0[1]",
                    "shape": [1, 2, 4],
                    "dtype": "torch.float16",
                },
                {
                    "name": "layer 1.state",
                    "shape": [5, 6],
                    "dtype": "torch.float32",
                },
            ],
        )

    def test_summarize_past_key_values_handles_dynamic_cache_layers(self):
        class FakeLinearAttentionLayer:
            def __init__(self):
                self.conv_states = torch.zeros(1, 4, 128, 4, dtype=torch.float16)
                self.recurrent_states = torch.zeros(1, 32, 128, 128, dtype=torch.float16)

        class FakeDynamicLayer:
            def __init__(self):
                self.keys = torch.zeros(1, 4, 1, 256, dtype=torch.float16)
                self.values = torch.zeros(1, 4, 1, 256, dtype=torch.float16)

        class FakeDynamicCache:
            def __init__(self):
                self.layers = [FakeLinearAttentionLayer(), FakeDynamicLayer()]

        summary = summarize_past_key_values(FakeDynamicCache())

        self.assertEqual(
            summary,
            [
                {
                    "name": "layer 0.conv_states",
                    "shape": [1, 4, 128, 4],
                    "dtype": "torch.float16",
                },
                {
                    "name": "layer 0.recurrent_states",
                    "shape": [1, 32, 128, 128],
                    "dtype": "torch.float16",
                },
                {
                    "name": "layer 1.keys",
                    "shape": [1, 4, 1, 256],
                    "dtype": "torch.float16",
                },
                {
                    "name": "layer 1.values",
                    "shape": [1, 4, 1, 256],
                    "dtype": "torch.float16",
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
