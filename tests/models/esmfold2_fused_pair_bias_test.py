"""Forward pair-bias addressing beyond the signed int32 element limit.

Run on CUDA with:
    pytest -o addopts='' tests/models/esmfold2_fused_pair_bias_test.py

Long cases need roughly 16 GiB of free device memory. A NaN-filled prefix
keeps the old kernel's wrapped negative offsets inside the allocation, making
the regression fail on nonfinite output instead of an illegal memory access.
"""

import math

import pytest
import torch
import torch.nn.functional as F


@pytest.mark.gpu
@pytest.mark.parametrize(
    "length",
    [
        65,
        pytest.param(1524, marks=pytest.mark.nightly),
        pytest.param(1956, marks=pytest.mark.nightly),
    ],
)
@torch.inference_mode()
def test_pair_bias_forward_large_batch_offsets(length):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("requires CUDA bfloat16 support")
    pytest.importorskip("triton")
    from esm.models.esmfold2.kernels.fused_attention_pair_bias import fused_pair_bias

    batch, channels, heads = 5, 256, 16
    prefix = 2**31 if length > 65 else 0
    elements = batch * length * length * channels
    output_elements = batch * heads * length * length
    required_bytes = 2 * (prefix + elements + output_elements) + 2**31
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < required_bytes:
        pytest.skip(f"requires {required_bytes / 2**30:.1f} GiB free GPU memory")

    generator = torch.Generator(device="cuda").manual_seed(20261010)
    storage = torch.empty(prefix + elements, device="cuda", dtype=torch.bfloat16)
    if prefix:
        storage[:prefix].fill_(float("nan"))
    z = storage[prefix:].view(batch, length, length, channels)
    z.normal_(generator=generator)
    weights = (
        torch.randn(heads, channels, device="cuda", generator=generator)
        / math.sqrt(channels)
    ).bfloat16()
    gamma = (
        1 + 0.1 * torch.randn(channels, device="cuda", generator=generator)
    ).bfloat16()
    beta = (
        0.1 * torch.randn(channels, device="cuda", generator=generator)
    ).bfloat16()
    mask = torch.ones((batch, length), device="cuda", dtype=torch.bool)
    mask[:, 7::13] = False

    actual = fused_pair_bias(z, mask, weights, gamma, beta, num_heads=heads)
    assert torch.isfinite(actual).all().item()

    # Check independent PyTorch LN + projection at both edges and the middle
    # of every batch, without allocating another full pair representation.
    queries = torch.tensor([0, length // 2, length - 1], device="cuda")
    normalized = F.layer_norm(
        z[:, queries].float(), (channels,), gamma.float(), beta.float(), 1e-5
    )
    reference = F.linear(normalized, weights.float()).permute(0, 3, 1, 2)
    reference = (reference + (~mask[:, None, None, :]) * -1e6).to(z.dtype)
    torch.testing.assert_close(
        actual[:, :, queries, :].float(), reference.float(), rtol=0.02, atol=0.02
    )
