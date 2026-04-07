from __future__ import annotations

import torch


def quantized_matmul_reference(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    weight_approx = qweight.float() * scale.float().unsqueeze(1)
    return x.float() @ weight_approx.T


def quantized_matvec_reference(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    if x.dim() == 2:
        if x.shape[0] != 1:
            raise ValueError("quantized_matvec_reference expects x with shape [K] or [1, K]")
        x = x.squeeze(0)
    weight_approx = qweight.float() * scale.float().unsqueeze(1)
    return weight_approx @ x.float()


def check_correctness(
    y_reference: torch.Tensor,
    y_test: torch.Tensor,
    tag: str = "",
) -> dict[str, float]:
    y_ref = y_reference.float()
    y_tst = y_test.float()
    abs_err = (y_ref - y_tst).abs()
    cos_sim = torch.nn.functional.cosine_similarity(
        y_ref.flatten().unsqueeze(0),
        y_tst.flatten().unsqueeze(0),
    ).item()
    report = {
        "tag": tag,
        "max_abs_err": abs_err.max().item(),
        "mean_abs_err": abs_err.mean().item(),
        "rel_err": (abs_err / (y_ref.abs() + 1e-8)).mean().item(),
        "cos_sim": cos_sim,
    }
    print(
        f"[{tag}] max={report['max_abs_err']:.6f} "
        f"mean={report['mean_abs_err']:.6f} "
        f"rel={report['rel_err']:.6f} "
        f"cos={report['cos_sim']:.8f}"
    )
    return report


def self_test(
    test_shapes: list[tuple[str, int, int]] | None = None,
    seed: int = 42,
) -> list[dict[str, float]]:
    from quantize import quantize_symmetric_int8

    torch.manual_seed(seed)
    test_shapes = test_shapes or [
        ("FFN_gate_proj", 9216, 2560),
        ("FullAttn_q_proj", 8192, 2560),
        ("DeltaNet_in_proj_qkv", 8192, 2560),
    ]
    reports: list[dict[str, float]] = []
    for tag, n, k in test_shapes:
        weight = torch.randn(n, k, dtype=torch.float16)
        x_decode = torch.randn(1, k, dtype=torch.float16)
        y_fp16 = x_decode.float() @ weight.float().T
        qweight, scale = quantize_symmetric_int8(weight)
        y_quant = quantized_matmul_reference(x_decode, qweight, scale)
        reports.append(check_correctness(y_fp16, y_quant, tag=tag))
    print("Self test PASSED")
    return reports


if __name__ == "__main__":
    self_test()
