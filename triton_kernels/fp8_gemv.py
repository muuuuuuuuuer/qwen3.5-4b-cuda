"""FP8 E4M3 GEMV Triton kernel for batch-1 decode inference.

Provides a decode-specialized FP8 GEMV kernel that reads FP8 weights,
dequantizes to FP32 with per-channel scales in SRAM, and computes
batch-1 matrix-vector product in a single kernel launch.

Reference:
    - FP8 E4M3 format (torch.float8_e4m3fn): 1 sign + 4 exponent + 3 mantissa
    - Range: +-448.0, 256 representable values per channel
    - Hardware target: NVIDIA Ada Lovelace+ (RTX 4090)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:
    triton = None
    tl = None


FP8_MAX: float = 448.0


@dataclass(frozen=True)
class FP8GEMVKernelConfig:
    bn: int = 64
    bk: int = 128
    num_warps: int = 4
    num_stages: int = 1


DEFAULT_FP8_GEMV_CONFIG = FP8GEMVKernelConfig()

FP8_GEMV_AUTOTUNE_CONFIGS = (
    FP8GEMVKernelConfig(bn=32, bk=64, num_warps=2, num_stages=1),
    FP8GEMVKernelConfig(bn=32, bk=128, num_warps=2, num_stages=1),
    FP8GEMVKernelConfig(bn=32, bk=128, num_warps=4, num_stages=1),
    FP8GEMVKernelConfig(bn=64, bk=64, num_warps=2, num_stages=1),
    DEFAULT_FP8_GEMV_CONFIG,
    FP8GEMVKernelConfig(bn=64, bk=128, num_warps=4, num_stages=2),
    FP8GEMVKernelConfig(bn=64, bk=128, num_warps=8, num_stages=1),
    FP8GEMVKernelConfig(bn=64, bk=256, num_warps=4, num_stages=1),
    FP8GEMVKernelConfig(bn=128, bk=64, num_warps=4, num_stages=1),
    FP8GEMVKernelConfig(bn=128, bk=128, num_warps=4, num_stages=1),
    FP8GEMVKernelConfig(bn=128, bk=128, num_warps=8, num_stages=1),
    FP8GEMVKernelConfig(bn=128, bk=256, num_warps=4, num_stages=1),
)

_AUTOTUNE_CACHE: dict[tuple[object, ...], FP8GEMVKernelConfig] = {}


def _normalize_kernel_config(
    kernel_config: FP8GEMVKernelConfig | None,
    n_dim: int,
    k_dim: int,
) -> FP8GEMVKernelConfig:
    config = kernel_config or DEFAULT_FP8_GEMV_CONFIG
    if config.bn <= 0 or config.bn > n_dim:
        raise ValueError("kernel_config.bn must be in the range [1, N]")
    if config.bk <= 0 or config.bk > k_dim:
        raise ValueError("kernel_config.bk must be in the range [1, K]")
    if config.num_warps <= 0:
        raise ValueError("kernel_config.num_warps must be positive")
    if config.num_stages <= 0:
        raise ValueError("kernel_config.num_stages must be positive")
    return config


def _autotune_cache_key(
    w: torch.Tensor,
    x: torch.Tensor,
    scale: torch.Tensor,
    n_dim: int,
    k_dim: int,
) -> tuple[object, ...]:
    return (
        "fp8_gemv",
        w.device,
        str(x.dtype),
        n_dim,
        k_dim,
    )


if triton is not None:

    _FP8_GEMV_TRITON_CONFIGS = [
        triton.Config(
            {"BN": config.bn, "BK": config.bk},
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )
        for config in FP8_GEMV_AUTOTUNE_CONFIGS
    ]

    @triton.autotune(
        configs=_FP8_GEMV_TRITON_CONFIGS,
        key=["N", "K"],
    )
    @triton.jit
    def _fp8_gemv_kernel_autotuned(
        w_ptr,
        scale_ptr,
        x_ptr,
        o_ptr,
        N: tl.constexpr,
        K: tl.constexpr,
        BK: tl.constexpr,
        BN: tl.constexpr,
    ):
        """Autotuned FP8 GEMV kernel. BK and BN are set by autotuner configs."""
        pid = tl.program_id(0)
        rows = pid * BN + tl.arange(0, BN)
        row_mask = rows < N

        scale = tl.load(scale_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

        acc = tl.zeros([BN], dtype=tl.float32)

        for k_start in range(0, K, BK):
            k_offs = k_start + tl.arange(0, BK)
            k_mask = k_offs < K

            x_val = tl.load(x_ptr + k_offs, mask=k_mask, other=0.0).to(tl.float32)

            w_offs = rows[:, None] * K + k_offs[None, :]
            w_mask = row_mask[:, None] & k_mask[None, :]
            w_val = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0).to(tl.float32)

            w_deq = w_val * scale[:, None]
            acc += tl.sum(w_deq * x_val[None, :], axis=1)

        tl.store(o_ptr + rows, acc.to(o_ptr.dtype.element_ty), mask=row_mask)

    @triton.jit
    def _fp8_gemv_kernel(
        w_ptr,
        scale_ptr,
        x_ptr,
        o_ptr,
        N: tl.constexpr,
        K: tl.constexpr,
        BK: tl.constexpr,
        BN: tl.constexpr,
    ):
        """Non-autotuned FP8 GEMV kernel. BK and BN are explicit launch params."""
        pid = tl.program_id(0)
        rows = pid * BN + tl.arange(0, BN)
        row_mask = rows < N

        scale = tl.load(scale_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

        acc = tl.zeros([BN], dtype=tl.float32)

        for k_start in range(0, K, BK):
            k_offs = k_start + tl.arange(0, BK)
            k_mask = k_offs < K

            x_val = tl.load(x_ptr + k_offs, mask=k_mask, other=0.0).to(tl.float32)

            w_offs = rows[:, None] * K + k_offs[None, :]
            w_mask = row_mask[:, None] & k_mask[None, :]
            w_val = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0).to(tl.float32)

            w_deq = w_val * scale[:, None]
            acc += tl.sum(w_deq * x_val[None, :], axis=1)

        tl.store(o_ptr + rows, acc.to(o_ptr.dtype.element_ty), mask=row_mask)


def fp8_gemv(
    weight_fp8: torch.Tensor,
    scale: torch.Tensor,
    x_fp16: torch.Tensor,
    kernel_config: FP8GEMVKernelConfig | None = None,
    use_autotune: bool = True,
) -> torch.Tensor:
    """FP8 GEMV: weight_fp8[N,K] @ x_fp16[K] → output_fp16[N] (batch-1 decode).

    Args:
        weight_fp8: FP8 E4M3 weight matrix of shape [N, K].
        scale: FP32 per-channel scale of shape [N, 1] or [N].
        x_fp16: FP16 activation vector of shape [K].
        kernel_config: Optional manual kernel configuration.
        use_autotune: Whether to use autotuned kernel (default True).

    Returns:
        FP16 output vector of shape [N].
    """
    if triton is None:
        raise RuntimeError("Triton is not available. Install triton to use FP8 GEMV kernel.")

    if not weight_fp8.is_cuda:
        raise ValueError("weight_fp8 must be a CUDA tensor")
    if not scale.is_cuda:
        raise ValueError("scale must be a CUDA tensor")
    if not x_fp16.is_cuda:
        raise ValueError("x_fp16 must be a CUDA tensor")

    if weight_fp8.ndim != 2:
        raise ValueError(f"weight_fp8 must be 2D, got {weight_fp8.ndim}D")
    if scale.ndim not in (1, 2):
        raise ValueError(f"scale must be 1D or 2D, got {scale.ndim}D")
    if x_fp16.ndim != 1:
        raise ValueError(f"x_fp16 must be 1D vector, got {x_fp16.ndim}D")

    N, K = weight_fp8.shape
    if x_fp16.shape[0] != K:
        raise ValueError(f"x_fp16 shape {x_fp16.shape} not compatible with weight shape [{N},{K}]")

    scale = scale.contiguous()
    if scale.ndim == 2:
        scale = scale.squeeze(-1)
    if scale.shape[0] != N:
        raise ValueError(f"scale shape {scale.shape} not compatible with weight N={N}")

    weight_fp8 = weight_fp8.contiguous()
    x_fp16 = x_fp16.contiguous()

    output = torch.empty(N, dtype=x_fp16.dtype, device=x_fp16.device)

    if use_autotune:
        cache_key = _autotune_cache_key(weight_fp8, x_fp16, scale, N, K)
        grid = lambda meta: (triton.cdiv(N, meta["BN"]),)

        _fp8_gemv_kernel_autotuned[grid](
            weight_fp8, scale, x_fp16, output,
            N=N, K=K,
        )

        if cache_key not in _AUTOTUNE_CACHE:
            best_cfg = getattr(_fp8_gemv_kernel_autotuned, "best_config", None)
            if best_cfg is not None and isinstance(best_cfg, triton.Config):
                _AUTOTUNE_CACHE[cache_key] = FP8GEMVKernelConfig(
                    bn=best_cfg.kwargs.get("BN", DEFAULT_FP8_GEMV_CONFIG.bn),
                    bk=best_cfg.kwargs.get("BK", DEFAULT_FP8_GEMV_CONFIG.bk),
                    num_warps=best_cfg.num_warps,
                    num_stages=best_cfg.num_stages,
                )
    else:
        config = _normalize_kernel_config(kernel_config, N, K)
        grid = (triton.cdiv(N, config.bn),)
        _fp8_gemv_kernel[grid](
            weight_fp8, scale, x_fp16, output,
            N=N, K=K, BK=config.bk, BN=config.bn,
            num_warps=config.num_warps, num_stages=config.num_stages,
        )

    return output


def fp8_gemv_reference(
    weight_fp8: torch.Tensor,
    scale: torch.Tensor,
    x_fp16: torch.Tensor,
) -> torch.Tensor:
    """PyTorch reference for FP8 GEMV correctness checking.

    Dequantizes FP8 weights to FP32, then computes FP32 matvec.
    Runs on the same device as inputs if CUDA, else CPU.

    Args:
        weight_fp8: FP8 weight [N, K].
        scale: FP32 scale [N] or [N, 1].
        x_fp16: FP16 activation [K].

    Returns:
        FP16 output [N].
    """
    w_fp32 = weight_fp8.float()
    if scale.ndim == 2:
        scale = scale.squeeze(-1)
    w_deq = w_fp32 * scale.float().unsqueeze(-1)

    x_fp32 = x_fp16.float()

    y_fp32 = w_deq @ x_fp32

    return y_fp32.to(x_fp16.dtype)


def fp8_gemv_cpu_reference(
    weight_fp8: torch.Tensor,
    scale: torch.Tensor,
    x_fp16: torch.Tensor,
) -> torch.Tensor:
    """Pure Python reference for offline/CPU correctness checking.

    Performs element-wise dequantization and dot product in double precision.
    """
    w = weight_fp8.float().double().cpu()
    if scale.ndim == 2:
        s = scale.squeeze(-1).float().double().cpu()
    else:
        s = scale.float().double().cpu()
    x = x_fp16.float().double().cpu()

    N, K = w.shape
    output = torch.zeros(N, dtype=torch.float64)
    for i in range(N):
        for j in range(K):
            output[i] += w[i, j] * s[i] * x[j]

    return output.to(x_fp16.dtype)
