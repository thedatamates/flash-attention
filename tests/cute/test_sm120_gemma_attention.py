from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from flash_attn.cute import flash_attn_func, flash_attn_varlen_func  # noqa: E402


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12,
    reason="SM120 attention tests require an SM120 CUDA device",
)


def _dense_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    window_size_left: int | None,
    window_size_right: int | None,
    dout: torch.Tensor | None = None,
):
    q_ref = q.detach().float().requires_grad_(dout is not None)
    k_ref = k.detach().float().requires_grad_(dout is not None)
    v_ref = v.detach().float().requires_grad_(dout is not None)
    _, seqlen, h_q, head_dim = q_ref.shape
    h_kv = k_ref.shape[2]
    q_t = q_ref.transpose(1, 2)
    k_t = k_ref.transpose(1, 2).repeat_interleave(h_q // h_kv, dim=1)
    v_t = v_ref.transpose(1, 2).repeat_interleave(h_q // h_kv, dim=1)
    scores = torch.matmul(q_t, k_t.transpose(-1, -2)) * (1.0 / math.sqrt(head_dim))
    q_idx = torch.arange(seqlen, device=q.device)[:, None]
    k_idx = torch.arange(seqlen, device=q.device)[None, :]
    mask = torch.ones((seqlen, seqlen), device=q.device, dtype=torch.bool)
    if causal:
        mask &= k_idx <= q_idx
    if window_size_left is not None:
        mask &= k_idx >= q_idx - window_size_left
    if window_size_right is not None:
        mask &= k_idx <= q_idx + window_size_right
    scores = scores.masked_fill(~mask[None, None], -torch.inf)
    out = torch.matmul(torch.softmax(scores, dim=-1), v_t).transpose(1, 2)
    lse = torch.logsumexp(scores, dim=-1)
    if dout is not None:
        out.backward(dout.float())
        return out.to(q.dtype), lse, q_ref.grad, k_ref.grad, v_ref.grad
    return out.to(q.dtype), lse


def _varlen_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    causal: bool,
    window_size_left: int | None,
    window_size_right: int | None,
):
    out = torch.empty_like(q)
    lse = torch.empty((q.shape[1], q.shape[0]), device=q.device, dtype=torch.float32)
    cu_cpu = cu_seqlens.detach().cpu().tolist()
    for start, end in zip(cu_cpu[:-1], cu_cpu[1:]):
        out_b, lse_b = _dense_reference(
            q[start:end].unsqueeze(0),
            k[start:end].unsqueeze(0),
            v[start:end].unsqueeze(0),
            causal=causal,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
        )
        out[start:end] = out_b.squeeze(0)
        lse[:, start:end] = lse_b.squeeze(0)
    return out, lse


def _dense_reference_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    window_size_left: int | None,
    window_size_right: int | None,
    dout: torch.Tensor | None = None,
    block_q: int = 64,
):
    q_ref = q.detach().float().requires_grad_(dout is not None)
    k_ref = k.detach().float().requires_grad_(dout is not None)
    v_ref = v.detach().float().requires_grad_(dout is not None)
    batch, seqlen, h_q, head_dim = q_ref.shape
    h_kv = k_ref.shape[2]
    gqa_ratio = h_q // h_kv
    out = torch.empty_like(q_ref)
    lse = torch.empty((batch, h_q, seqlen), device=q.device, dtype=torch.float32)
    for q_start in range(0, seqlen, block_q):
        q_end = min(seqlen, q_start + block_q)
        k_start = 0 if window_size_left is None else max(0, q_start - int(window_size_left))
        k_end = seqlen if not causal else q_end
        if window_size_right is not None:
            k_end = min(seqlen, k_end + int(window_size_right))
        q_blk = q_ref[:, q_start:q_end]
        k_blk = k_ref[:, k_start:k_end].transpose(1, 2).repeat_interleave(gqa_ratio, dim=1)
        v_blk = v_ref[:, k_start:k_end].transpose(1, 2).repeat_interleave(gqa_ratio, dim=1)
        scores = torch.matmul(q_blk.transpose(1, 2), k_blk.transpose(-1, -2)) * (1.0 / math.sqrt(head_dim))
        q_idx = torch.arange(q_start, q_end, device=q.device)[:, None]
        k_idx = torch.arange(k_start, k_end, device=q.device)[None, :]
        mask = torch.ones((q_end - q_start, k_end - k_start), device=q.device, dtype=torch.bool)
        if causal:
            mask &= k_idx <= q_idx
        if window_size_left is not None:
            mask &= k_idx >= q_idx - int(window_size_left)
        if window_size_right is not None:
            mask &= k_idx <= q_idx + int(window_size_right)
        scores = scores.masked_fill(~mask[None, None], -torch.inf)
        out_blk = torch.matmul(torch.softmax(scores, dim=-1), v_blk).transpose(1, 2)
        out[:, q_start:q_end] = out_blk
        lse[:, :, q_start:q_end] = torch.logsumexp(scores, dim=-1)
        if dout is not None:
            out_blk.backward(dout[:, q_start:q_end].float())
    if dout is not None:
        return out.to(q.dtype), lse, q_ref.grad, k_ref.grad, v_ref.grad
    return out.to(q.dtype), lse


def _min_row_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.float()
    b_f = b.float()
    cos = torch.nn.functional.cosine_similarity(a_f, b_f, dim=-1)
    both_zero = (a_f.norm(dim=-1) == 0) & (b_f.norm(dim=-1) == 0)
    cos = torch.where(both_zero, torch.ones_like(cos), cos)
    return float(cos.min().item())


def _run_dense_once(q0, k0, v0, dout, *, window_size_left, window_size_right):
    q = q0.detach().clone().requires_grad_(True)
    k = k0.detach().clone().requires_grad_(True)
    v = v0.detach().clone().requires_grad_(True)
    out, lse = flash_attn_func(
        q,
        k,
        v,
        causal=True,
        softmax_scale=1.0 / math.sqrt(q.shape[-1]),
        window_size_left=window_size_left,
        window_size_right=window_size_right,
    )
    torch.cuda.synchronize()
    out.backward(dout)
    torch.cuda.synchronize()
    return out.detach(), lse.detach(), q.grad.detach(), k.grad.detach(), v.grad.detach()


@pytest.mark.parametrize(
    ("name", "shape", "window_size_left", "window_size_right"),
    [
        ("swa_small", (1, 128, 32, 16, 256), 64, 0),
        ("global_small", (1, 128, 32, 4, 512), None, None),
    ],
)
def test_dense_small_forward_backward(name, shape, window_size_left, window_size_right):
    del name
    batch, seqlen, h_q, h_kv, head_dim = shape
    torch.manual_seed(0)
    q = torch.randn(batch, seqlen, h_q, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(batch, seqlen, h_kv, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(batch, seqlen, h_kv, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dout = torch.randn_like(q)

    out, lse = flash_attn_func(
        q,
        k,
        v,
        causal=True,
        softmax_scale=1.0 / math.sqrt(head_dim),
        window_size_left=window_size_left,
        window_size_right=window_size_right,
    )
    torch.cuda.synchronize()
    out.backward(dout)
    out_ref, lse_ref, dq_ref, dk_ref, dv_ref = _dense_reference(
        q,
        k,
        v,
        causal=True,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        dout=dout,
    )
    torch.cuda.synchronize()

    assert _min_row_cos(out, out_ref) >= 0.999
    assert _min_row_cos(q.grad, dq_ref) >= 0.999
    assert _min_row_cos(k.grad, dk_ref) >= 0.999
    assert _min_row_cos(v.grad, dv_ref) >= 0.999
    assert float((lse - lse_ref).abs().max().item()) <= 1e-2


@pytest.mark.parametrize(
    ("shape", "window_size_left", "window_size_right"),
    [
        ((1, 256, 32, 16, 256), 64, 0),
        ((1, 256, 32, 4, 512), None, None),
    ],
)
def test_dense_medium_forward_backward(shape, window_size_left, window_size_right):
    batch, seqlen, h_q, h_kv, head_dim = shape
    torch.manual_seed(10)
    q = torch.randn(batch, seqlen, h_q, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(batch, seqlen, h_kv, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(batch, seqlen, h_kv, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dout = torch.randn_like(q)
    out, lse = flash_attn_func(
        q,
        k,
        v,
        causal=True,
        softmax_scale=1.0 / math.sqrt(head_dim),
        window_size_left=window_size_left,
        window_size_right=window_size_right,
    )
    torch.cuda.synchronize()
    out.backward(dout)
    out_ref, lse_ref, dq_ref, dk_ref, dv_ref = _dense_reference(
        q,
        k,
        v,
        causal=True,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        dout=dout,
    )
    torch.cuda.synchronize()
    assert _min_row_cos(out, out_ref) >= 0.999
    assert _min_row_cos(q.grad, dq_ref) >= 0.999
    assert _min_row_cos(k.grad, dk_ref) >= 0.999
    assert _min_row_cos(v.grad, dv_ref) >= 0.999
    assert float((lse - lse_ref).abs().max().item()) <= 1e-2


@pytest.mark.parametrize(
    ("shape", "window_size_left", "window_size_right"),
    [
        ((1, 128, 32, 16, 256), 64, 0),
        ((1, 128, 32, 4, 512), None, None),
    ],
)
def test_dense_deterministic_repeated_invocation(shape, window_size_left, window_size_right):
    batch, seqlen, h_q, h_kv, head_dim = shape
    torch.manual_seed(12)
    q = torch.randn(batch, seqlen, h_q, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, seqlen, h_kv, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, seqlen, h_kv, head_dim, device="cuda", dtype=torch.bfloat16)
    dout = torch.randn_like(q)

    first = _run_dense_once(
        q,
        k,
        v,
        dout,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
    )
    second = _run_dense_once(
        q,
        k,
        v,
        dout,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
    )
    for actual, expected in zip(first, second):
        assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    ("shape", "window_size_left", "window_size_right", "seed", "coords"),
    [
        (
            (1, 16, 32, 16, 256),
            1024,
            0,
            21,
            {
                "q": (0, 11, 9, 153),
                "k": (0, 11, 9, 153),
                "v": (0, 2, 0, 0),
            },
        ),
        (
            (1, 16, 32, 4, 512),
            None,
            None,
            22,
            {
                "q": (0, 4, 12, 348),
                "k": (0, 3, 1, 97),
                "v": (0, 2, 0, 0),
            },
        ),
    ],
)
def test_dense_numerical_gradient_spotcheck(shape, window_size_left, window_size_right, seed, coords):
    batch, seqlen, h_q, h_kv, head_dim = shape
    torch.manual_seed(seed)
    q = torch.randn(batch, seqlen, h_q, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(batch, seqlen, h_kv, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(batch, seqlen, h_kv, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn_like(q).float()

    def loss(q_arg, k_arg, v_arg):
        out, _ = flash_attn_func(
            q_arg,
            k_arg,
            v_arg,
            causal=True,
            softmax_scale=1.0 / math.sqrt(head_dim),
            window_size_left=window_size_left,
            window_size_right=window_size_right,
        )
        torch.cuda.synchronize()
        return (out.float() * weight).sum()

    loss(q, k, v).backward()
    torch.cuda.synchronize()
    eps = 1.0
    for name, tensor, grad in (("q", q, q.grad), ("k", k, k.grad), ("v", v, v.grad)):
        idx = coords[name]
        plus = tensor.detach().clone()
        minus = tensor.detach().clone()
        plus[idx] = plus[idx] + eps
        minus[idx] = minus[idx] - eps
        args = {"q": q.detach(), "k": k.detach(), "v": v.detach()}
        args[name] = plus
        loss_plus = loss(args["q"], args["k"], args["v"])
        args[name] = minus
        loss_minus = loss(args["q"], args["k"], args["v"])
        finite_diff = ((loss_plus - loss_minus) / (2.0 * eps)).item()
        analytic = grad[idx].float().item()
        rel_err = abs(finite_diff - analytic) / max(1e-6, abs(finite_diff), abs(analytic))
        assert rel_err <= 1e-2


@pytest.mark.parametrize(
    ("shape", "window_size_left", "window_size_right"),
    [
        ((1, 8192, 32, 16, 256), 1024, 0),
        ((1, 16384, 32, 16, 256), 1024, 0),
        ((1, 8192, 32, 4, 512), None, None),
    ],
)
def test_dense_production_matrix_forward_backward(shape, window_size_left, window_size_right):
    batch, seqlen, h_q, h_kv, head_dim = shape
    torch.manual_seed(13)
    q = torch.randn(batch, seqlen, h_q, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(batch, seqlen, h_kv, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(batch, seqlen, h_kv, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dout = torch.randn_like(q)
    out, lse = flash_attn_func(
        q,
        k,
        v,
        causal=True,
        softmax_scale=1.0 / math.sqrt(head_dim),
        window_size_left=window_size_left,
        window_size_right=window_size_right,
    )
    torch.cuda.synchronize()
    lse_before_backward = lse.detach().clone()
    out.backward(dout)
    torch.cuda.synchronize()
    out_ref, lse_ref, dq_ref, dk_ref, dv_ref = _dense_reference_chunked(
        q,
        k,
        v,
        causal=True,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        dout=dout,
    )
    torch.cuda.synchronize()
    assert _min_row_cos(out, out_ref) >= 0.999
    assert _min_row_cos(q.grad, dq_ref) >= 0.999
    assert _min_row_cos(k.grad, dk_ref) >= 0.999
    assert _min_row_cos(v.grad, dv_ref) >= 0.999
    assert float((lse_before_backward - lse_ref).abs().max().item()) <= 1e-2
    if shape == (1, 8192, 32, 16, 256):
        from gemma_triton_flash_attn import attention_flash_gqa

        out_triton = attention_flash_gqa(
            q.detach().transpose(1, 2).contiguous(),
            k.detach().transpose(1, 2).contiguous(),
            v.detach().transpose(1, 2).contiguous(),
            causal=True,
            slide_size=window_size_left + 1,
        ).transpose(1, 2)
        torch.cuda.synchronize()
        assert _min_row_cos(out, out_triton) >= 0.999


def test_varlen_pack_forward_exact_spec_case():
    import gc

    # CuTe's process-local JIT cache can reuse stale SM120 dense-specialization
    # state across dense -> varlen tests in the same pytest worker. Keep this
    # test isolated until the upstream cache-key/state issue is fixed.
    from flash_attn.cute.interface import _flash_attn_fwd

    _flash_attn_fwd.compile_cache.clear()
    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(1)
    cu_seqlens = torch.tensor([0, 512, 1280, 2048], device="cuda", dtype=torch.int32)
    total, h_q, h_kv, head_dim = 2048, 32, 16, 256
    q = torch.randn(total, h_q, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(total, h_kv, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(total, h_kv, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dout = torch.randn_like(q)

    out, lse = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=768,
        max_seqlen_k=768,
        causal=True,
        softmax_scale=1.0 / math.sqrt(head_dim),
        window_size_left=1024,
        window_size_right=0,
    )
    torch.cuda.synchronize()
    out_ref, lse_ref = _varlen_reference(
        q,
        k,
        v,
        cu_seqlens,
        causal=True,
        window_size_left=1024,
        window_size_right=0,
    )
    torch.cuda.synchronize()

    assert _min_row_cos(out, out_ref) >= 0.999
    assert float((lse - lse_ref).abs().max().item()) <= 1e-2

    out.backward(dout)
    dq_ref = torch.empty_like(q, dtype=torch.float32)
    dk_ref = torch.empty_like(k, dtype=torch.float32)
    dv_ref = torch.empty_like(v, dtype=torch.float32)
    cu_cpu = cu_seqlens.detach().cpu().tolist()
    for start, end in zip(cu_cpu[:-1], cu_cpu[1:]):
        _, _, dq_b, dk_b, dv_b = _dense_reference(
            q[start:end].detach().unsqueeze(0),
            k[start:end].detach().unsqueeze(0),
            v[start:end].detach().unsqueeze(0),
            causal=True,
            window_size_left=1024,
            window_size_right=0,
            dout=dout[start:end].detach().unsqueeze(0),
        )
        dq_ref[start:end] = dq_b.squeeze(0)
        dk_ref[start:end] = dk_b.squeeze(0)
        dv_ref[start:end] = dv_b.squeeze(0)
    torch.cuda.synchronize()

    assert _min_row_cos(q.grad, dq_ref) >= 0.999
    assert _min_row_cos(k.grad, dk_ref) >= 0.999
    assert _min_row_cos(v.grad, dv_ref) >= 0.999


@pytest.mark.parametrize(
    ("shape", "window_size_left", "window_size_right"),
    [
        ((1, 130, 32, 16, 256), 512, 0),
        ((1, 130, 32, 4, 512), None, None),
    ],
)
def test_dense_edge_ragged_and_window_larger_than_sequence(shape, window_size_left, window_size_right):
    batch, seqlen, h_q, h_kv, head_dim = shape
    torch.manual_seed(3)
    q = torch.randn(batch, seqlen, h_q, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(batch, seqlen, h_kv, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(batch, seqlen, h_kv, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out, lse = flash_attn_func(
        q,
        k,
        v,
        causal=True,
        softmax_scale=1.0 / math.sqrt(head_dim),
        window_size_left=window_size_left,
        window_size_right=window_size_right,
    )
    torch.cuda.synchronize()
    out.float().sum().backward()
    torch.cuda.synchronize()
    assert torch.isfinite(out.float()).all()
    assert torch.isfinite(lse).all()
    assert torch.isfinite(q.grad.float()).all()
    assert torch.isfinite(k.grad.float()).all()
    assert torch.isfinite(v.grad.float()).all()


def test_varlen_zero_length_sequence_edge_case():
    torch.manual_seed(4)
    cu_seqlens = torch.tensor([0, 16, 16, 32], device="cuda", dtype=torch.int32)
    total, h_q, h_kv, head_dim = 32, 32, 16, 256
    q = torch.randn(total, h_q, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(total, h_kv, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(total, h_kv, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out, lse = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=16,
        max_seqlen_k=16,
        causal=True,
        softmax_scale=1.0 / math.sqrt(head_dim),
        window_size_left=64,
        window_size_right=0,
    )
    torch.cuda.synchronize()
    out.float().sum().backward()
    torch.cuda.synchronize()
    assert torch.isfinite(out.float()).all()
    assert torch.isfinite(lse).all()
    assert torch.isfinite(q.grad.float()).all()
    assert torch.isfinite(k.grad.float()).all()
    assert torch.isfinite(v.grad.float()).all()


def test_varlen_all_padding_rows_edge_case():
    torch.manual_seed(5)
    cu_seqlens_q = torch.tensor([0, 4, 8], device="cuda", dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, 0, 4], device="cuda", dtype=torch.int32)
    h_q, h_kv, head_dim = 32, 16, 256
    q = torch.randn(8, h_q, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(4, h_kv, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(4, h_kv, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out, lse = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=4,
        max_seqlen_k=4,
        causal=True,
        softmax_scale=1.0 / math.sqrt(head_dim),
        window_size_left=64,
        window_size_right=0,
    )
    torch.cuda.synchronize()
    out.float().sum().backward()
    torch.cuda.synchronize()
    assert torch.equal(out[:4], torch.zeros_like(out[:4]))
    assert torch.isneginf(lse[:, :4]).all()
    assert torch.equal(q.grad[:4], torch.zeros_like(q.grad[:4]))
    assert torch.isfinite(out[4:].float()).all()
    assert torch.isfinite(lse[:, 4:]).all()
    assert torch.isfinite(q.grad[4:].float()).all()
    assert torch.isfinite(k.grad.float()).all()
    assert torch.isfinite(v.grad.float()).all()
