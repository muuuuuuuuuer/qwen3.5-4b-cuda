import types
import unittest

import torch

from triton_kernels.deltanet_decode import deltanet_decode_reference
from triton_kernels.qwen35_integration import (
    apply_qwen35_deltanet_triton_patch,
    configure_qwen35_deltanet_runtime,
    describe_qwen35_deltanet_runtime,
    get_qwen35_triton_patch_stats,
    qwen35_triton_decode_forward_base,
    qwen35_triton_decode_forward,
    qwen35_triton_recurrent_gated_delta_rule,
    reset_qwen35_triton_patch_stats,
)


def make_recurrent_inputs(batch_size: int = 2, device: str = "cuda"):
    torch.manual_seed(0)
    query = torch.randn(batch_size, 1, 32, 128, dtype=torch.float16, device=device)
    key = torch.randn(batch_size, 1, 32, 128, dtype=torch.float16, device=device)
    value = torch.randn(batch_size, 1, 32, 128, dtype=torch.float16, device=device)
    g = -torch.rand(batch_size, 1, 32, dtype=torch.float32, device=device)
    beta = torch.sigmoid(torch.randn(batch_size, 1, 32, dtype=torch.float32, device=device))
    initial_state = torch.randn(batch_size, 32, 128, 128, dtype=torch.float32, device=device) * 0.01
    return query, key, value, g, beta, initial_state


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Qwen integration tests")
class Qwen35RecurrentAdapterTests(unittest.TestCase):
    def test_recurrent_adapter_matches_reference(self):
        reset_qwen35_triton_patch_stats()
        query, key, value, g, beta, initial_state = make_recurrent_inputs()
        expected_state = initial_state.clone()
        expected_output = torch.empty_like(value)
        for batch_idx in range(query.shape[0]):
            expected_output[batch_idx, 0] = deltanet_decode_reference(
                query[batch_idx, 0],
                key[batch_idx, 0],
                value[batch_idx, 0],
                g[batch_idx, 0],
                beta[batch_idx, 0],
                expected_state[batch_idx],
            ).to(value.dtype)

        output, final_state = qwen35_triton_recurrent_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=initial_state.clone(),
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )

        self.assertLess((expected_output.float() - output.float()).abs().max().item(), 1e-3)
        self.assertLess((expected_state - final_state).abs().max().item(), 1e-3)
        self.assertEqual(get_qwen35_triton_patch_stats()["calls"], 1)
        self.assertEqual(get_qwen35_triton_patch_stats()["batch_items"], query.shape[0])
        self.assertEqual(get_qwen35_triton_patch_stats()["tokens"], query.shape[0])


class Qwen35PatchTests(unittest.TestCase):
    def test_patch_replaces_linear_attention_recurrent_function(self):
        class FakeLinearAttn:
            def __init__(self):
                self.recurrent_gated_delta_rule = object()
                self.forward = types.MethodType(lambda self, *args, **kwargs: None, self)

        class FakeLayer:
            def __init__(self):
                self.linear_attn = FakeLinearAttn()

        class FakeLanguageModel:
            def __init__(self):
                self.layers = [FakeLayer(), types.SimpleNamespace(), FakeLayer()]

        class FakeWrappedModel:
            def __init__(self):
                self.language_model = FakeLanguageModel()

        class FakeRoot:
            def __init__(self):
                self.model = FakeWrappedModel()

        model = FakeRoot()

        patched = apply_qwen35_deltanet_triton_patch(model)

        self.assertEqual(patched, 2)
        self.assertIs(
            model.model.language_model.layers[0].linear_attn.recurrent_gated_delta_rule,
            qwen35_triton_recurrent_gated_delta_rule,
        )
        self.assertIs(
            model.model.language_model.layers[2].linear_attn.recurrent_gated_delta_rule,
            qwen35_triton_recurrent_gated_delta_rule,
        )

    def test_full_forward_patch_replaces_forward_and_keeps_original(self):
        class FakeLinearAttn:
            def __init__(self):
                self.recurrent_gated_delta_rule = object()
                self.forward = types.MethodType(lambda self, *args, **kwargs: "orig", self)

        class FakeLayer:
            def __init__(self):
                self.linear_attn = FakeLinearAttn()

        class FakeLanguageModel:
            def __init__(self):
                self.layers = [FakeLayer(), FakeLayer()]

        class FakeWrappedModel:
            def __init__(self):
                self.language_model = FakeLanguageModel()

        class FakeRoot:
            def __init__(self):
                self.model = FakeWrappedModel()

        model = FakeRoot()
        original_forward = model.model.language_model.layers[0].linear_attn.forward

        patched = apply_qwen35_deltanet_triton_patch(model, patch_mode="full_forward")

        self.assertEqual(patched, 2)
        self.assertIs(
            model.model.language_model.layers[0].linear_attn.forward.__func__,
            qwen35_triton_decode_forward,
        )
        self.assertIs(model.model.language_model.layers[0].linear_attn._original_forward, original_forward)

    def test_configure_runtime_torch_restores_original_forward(self):
        class FakeLinearAttn:
            def __init__(self):
                self.recurrent_gated_delta_rule = object()
                self.forward = types.MethodType(lambda self, *args, **kwargs: "orig", self)

        class FakeLayer:
            def __init__(self):
                self.linear_attn = FakeLinearAttn()

        class FakeLanguageModel:
            def __init__(self):
                self.layers = [FakeLayer()]

        class FakeWrappedModel:
            def __init__(self):
                self.language_model = FakeLanguageModel()

        class FakeRoot:
            def __init__(self):
                self.model = FakeWrappedModel()

        model = FakeRoot()
        original_forward = model.model.language_model.layers[0].linear_attn.forward
        apply_qwen35_deltanet_triton_patch(model, patch_mode="full_forward")

        configured = configure_qwen35_deltanet_runtime(model, mode="torch")

        self.assertEqual(configured, 1)
        self.assertIs(model.model.language_model.layers[0].linear_attn.forward, original_forward)

    def test_configure_runtime_triton_base_uses_base_forward(self):
        class FakeLinearAttn:
            def __init__(self):
                self.recurrent_gated_delta_rule = object()
                self.forward = types.MethodType(lambda self, *args, **kwargs: "orig", self)
                self.causal_conv1d_update = lambda *args, **kwargs: None
                self.chunk_gated_delta_rule = lambda *args, **kwargs: None

        class FakeLayer:
            def __init__(self):
                self.linear_attn = FakeLinearAttn()

        class FakeLanguageModel:
            def __init__(self):
                self.layers = [FakeLayer()]

        class FakeWrappedModel:
            def __init__(self):
                self.language_model = FakeLanguageModel()

        class FakeRoot:
            def __init__(self):
                self.model = FakeWrappedModel()

        model = FakeRoot()

        configured = configure_qwen35_deltanet_runtime(model, mode="triton_base")

        self.assertEqual(configured, 1)
        self.assertIs(
            model.model.language_model.layers[0].linear_attn.forward.__func__,
            qwen35_triton_decode_forward_base,
        )

    def test_describe_runtime_reports_mode_specific_forward_impl(self):
        class FakeLinearAttn:
            def __init__(self):
                self.recurrent_gated_delta_rule = object()
                self.forward = types.MethodType(lambda self, *args, **kwargs: "orig", self)
                self.causal_conv1d_update = lambda *args, **kwargs: None
                self.chunk_gated_delta_rule = lambda *args, **kwargs: None

        class FakeLayer:
            def __init__(self):
                self.linear_attn = FakeLinearAttn()

        class FakeLanguageModel:
            def __init__(self):
                self.layers = [FakeLayer()]

        class FakeWrappedModel:
            def __init__(self):
                self.language_model = FakeLanguageModel()

        class FakeRoot:
            def __init__(self):
                self.model = FakeWrappedModel()

        model = FakeRoot()
        configure_qwen35_deltanet_runtime(model, mode="triton_base")

        description = describe_qwen35_deltanet_runtime(model)

        self.assertEqual(description["forward_impl"], "qwen35_triton_decode_forward_base")

    def test_configure_runtime_triton_alias_uses_fused_forward(self):
        class FakeLinearAttn:
            def __init__(self):
                self.recurrent_gated_delta_rule = object()
                self.forward = types.MethodType(lambda self, *args, **kwargs: "orig", self)
                self.causal_conv1d_update = lambda *args, **kwargs: None
                self.chunk_gated_delta_rule = lambda *args, **kwargs: None

        class FakeLayer:
            def __init__(self):
                self.linear_attn = FakeLinearAttn()

        class FakeLanguageModel:
            def __init__(self):
                self.layers = [FakeLayer()]

        class FakeWrappedModel:
            def __init__(self):
                self.language_model = FakeLanguageModel()

        class FakeRoot:
            def __init__(self):
                self.model = FakeWrappedModel()

        model = FakeRoot()

        configured = configure_qwen35_deltanet_runtime(model, mode="triton")

        self.assertEqual(configured, 1)
        self.assertIs(
            model.model.language_model.layers[0].linear_attn.forward.__func__,
            qwen35_triton_decode_forward,
        )


if __name__ == "__main__":
    unittest.main()
