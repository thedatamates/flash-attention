import math
from typing import Optional

import torch
import triton
import triton.language as tl


def _default_scale(head_dim: int) -> float:
    return 1.0 / math.sqrt(head_dim)


def _is_default_scale(softmax_scale: Optional[float], head_dim: int) -> bool:
    return softmax_scale is None or math.isclose(
        float(softmax_scale), _default_scale(head_dim), rel_tol=0.0, abs_tol=1e-12
    )


def _device_arch(t: torch.Tensor) -> int:
    if not t.is_cuda:
        return 0
    index = torch.cuda.current_device() if t.device.index is None else t.device.index
    major, minor = torch.cuda.get_device_capability(index)
    return major * 10 + minor


def _slide_size(window_size_left: Optional[int], window_size_right: Optional[int]) -> int:
    if window_size_left is None and window_size_right is None:
        return 0
    if window_size_right not in (None, 0):
        raise NotImplementedError("SM120 Gemma wrapper only supports causal right-window 0")
    # gemma-triton-flash-attn uses i - j < slide_size. CuTe/FA window_left is inclusive.
    return int(window_size_left or 0) + 1


@triton.jit
def _sm120_gemma_lse_kernel(
    Q,
    K,
    LSE,
    stride_qb,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_lseb,
    stride_lseh,
    stride_lsen,
    N_Q_HEADS: tl.constexpr,
    N_KV_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    scale,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SLIDE_SIZE: tl.constexpr,
):
    q_block_idx = tl.program_id(0)
    bh_idx = tl.program_id(1)
    q_h_idx = bh_idx % N_Q_HEADS
    b_idx = bh_idx // N_Q_HEADS
    kv_h_idx = q_h_idx * N_KV_HEADS // N_Q_HEADS

    q_offsets = q_block_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = q_offsets < SEQ_LEN
    d = tl.arange(0, HEAD_DIM)
    q = tl.load(
        Q + b_idx * stride_qb + q_offsets[:, None] * stride_qn + q_h_idx * stride_qh + d[None, :] * stride_qd,
        mask=q_mask[:, None],
        other=0.0,
    )

    if IS_CAUSAL:
        kv_end = (q_block_idx + 1) * BLOCK_Q
    else:
        kv_end = SEQ_LEN
    if IS_CAUSAL and SLIDE_SIZE > 0:
        kv_min = tl.maximum(0, q_block_idx * BLOCK_Q - SLIDE_SIZE + 1)
        kv_loop_start = (kv_min // BLOCK_KV) * BLOCK_KV
    else:
        kv_loop_start = 0

    log2e: tl.constexpr = 1.4426950408889634
    scale_log2e = scale * log2e
    row_max = tl.full([BLOCK_Q], -float("inf"), dtype=tl.float32)
    row_sum = tl.zeros([BLOCK_Q], dtype=tl.float32)
    for kv_start in range(kv_loop_start, kv_end, BLOCK_KV):
        kv_offsets = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask = kv_offsets < SEQ_LEN
        k = tl.load(
            K + b_idx * stride_kb + kv_offsets[:, None] * stride_kn + kv_h_idx * stride_kh + d[None, :] * stride_kd,
            mask=kv_mask[:, None],
            other=0.0,
        )
        scores = tl.dot(q, tl.trans(k)).to(tl.float32) * scale_log2e
        if IS_CAUSAL:
            if SLIDE_SIZE > 0:
                valid = (kv_offsets[None, :] <= q_offsets[:, None]) & (
                    q_offsets[:, None] - kv_offsets[None, :] < SLIDE_SIZE
                ) & kv_mask[None, :]
            else:
                valid = (kv_offsets[None, :] <= q_offsets[:, None]) & kv_mask[None, :]
        else:
            valid = kv_mask[None, :]
        scores = tl.where(valid, scores, -float("inf"))
        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(row_max, block_max)
        row_sum = row_sum * tl.math.exp2(row_max - new_max) + tl.sum(tl.math.exp2(scores - new_max[:, None]), axis=1)
        row_max = new_max

    lse = row_max / log2e + tl.log(row_sum)
    tl.store(
        LSE + b_idx * stride_lseb + q_h_idx * stride_lseh + q_offsets * stride_lsen,
        lse,
        mask=q_mask,
    )


def _triton_dense_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    causal: bool,
    window_size_left: Optional[int],
    window_size_right: Optional[int],
) -> torch.Tensor:
    batch, seqlen, h_q, head_dim = q.shape
    h_kv = k.shape[2]
    lse = torch.empty((batch, h_q, seqlen), device=q.device, dtype=torch.float32)
    slide_size = _slide_size(window_size_left, window_size_right)
    if slide_size >= seqlen:
        slide_size = 0
    block_q = 16 if head_dim <= 256 else 8
    block_kv = 16
    _sm120_gemma_lse_kernel[(triton.cdiv(seqlen, block_q), batch * h_q)](
        q,
        k,
        lse,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        N_Q_HEADS=h_q,
        N_KV_HEADS=h_kv,
        SEQ_LEN=seqlen,
        HEAD_DIM=head_dim,
        scale=_default_scale(head_dim),
        BLOCK_Q=block_q,
        BLOCK_KV=block_kv,
        IS_CAUSAL=causal,
        SLIDE_SIZE=slide_size,
        num_warps=8,
        num_stages=1,
    )
    return lse


@triton.jit
def _sm120_gemma_delta_kernel(
    Q,
    K,
    V,
    dO,
    LSE,
    Delta,
    stride_qb,
    stride_qh,
    stride_qn,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_dob,
    stride_doh,
    stride_don,
    stride_dod,
    stride_lseb,
    stride_lseh,
    stride_lsen,
    stride_db,
    stride_dh,
    stride_dn,
    N_Q_HEADS: tl.constexpr,
    N_KV_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    scale,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SLIDE_SIZE: tl.constexpr,
):
    q_block_idx = tl.program_id(0)
    bh_idx = tl.program_id(1)
    q_h_idx = bh_idx % N_Q_HEADS
    b_idx = bh_idx // N_Q_HEADS
    kv_h_idx = q_h_idx * N_KV_HEADS // N_Q_HEADS

    q_offsets = q_block_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = q_offsets < SEQ_LEN
    d = tl.arange(0, HEAD_DIM)

    q_base = Q + b_idx * stride_qb + q_h_idx * stride_qh
    k_base = K + b_idx * stride_kb + kv_h_idx * stride_kh
    v_base = V + b_idx * stride_vb + kv_h_idx * stride_vh
    do_base = dO + b_idx * stride_dob + q_h_idx * stride_doh

    q = tl.load(q_base + q_offsets[:, None] * stride_qn + d[None, :] * stride_qd, mask=q_mask[:, None], other=0.0)
    do = tl.load(do_base + q_offsets[:, None] * stride_don + d[None, :] * stride_dod, mask=q_mask[:, None], other=0.0)

    lse = tl.load(
        LSE + b_idx * stride_lseb + q_h_idx * stride_lseh + q_offsets * stride_lsen,
        mask=q_mask,
        other=0.0,
    )

    if IS_CAUSAL:
        kv_end = (q_block_idx + 1) * BLOCK_Q
    else:
        kv_end = SEQ_LEN
    if IS_CAUSAL and SLIDE_SIZE > 0:
        kv_min = tl.maximum(0, q_block_idx * BLOCK_Q - SLIDE_SIZE + 1)
        kv_loop_start = (kv_min // BLOCK_KV) * BLOCK_KV
    else:
        kv_loop_start = 0

    log2e: tl.constexpr = 1.4426950408889634
    scale_log2e = scale * log2e
    lse_log2 = lse * log2e

    delta = tl.zeros([BLOCK_Q], dtype=tl.float32)
    for kv_start in range(kv_loop_start, kv_end, BLOCK_KV):
        kv_offsets = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask = kv_offsets < SEQ_LEN
        k = tl.load(k_base + kv_offsets[:, None] * stride_kn + d[None, :] * stride_kd, mask=kv_mask[:, None], other=0.0)
        v = tl.load(v_base + kv_offsets[:, None] * stride_vn + d[None, :] * stride_vd, mask=kv_mask[:, None], other=0.0)
        scores = tl.dot(q.to(tl.float32), tl.trans(k.to(tl.float32)), input_precision="ieee").to(tl.float32) * scale_log2e
        if IS_CAUSAL:
            if SLIDE_SIZE > 0:
                valid = (kv_offsets[None, :] <= q_offsets[:, None]) & (
                    q_offsets[:, None] - kv_offsets[None, :] < SLIDE_SIZE
                ) & kv_mask[None, :]
            else:
                valid = (kv_offsets[None, :] <= q_offsets[:, None]) & kv_mask[None, :]
        else:
            valid = kv_mask[None, :]
        scores = tl.where(valid, scores, -float("inf"))
        p = tl.math.exp2(scores - lse_log2[:, None])
        dp = tl.dot(do.to(tl.float32), tl.trans(v.to(tl.float32)), input_precision="ieee").to(tl.float32)
        delta += tl.sum(tl.where(valid, p * dp, 0.0), axis=1)

    tl.store(
        Delta + b_idx * stride_db + q_h_idx * stride_dh + q_offsets * stride_dn,
        delta,
        mask=q_mask,
    )


@triton.jit
def _sm120_gemma_bwd_dq_kernel(
    Q,
    K,
    V,
    dO,
    O,
    dQ,
    LSE,
    Delta,
    stride_qb,
    stride_qh,
    stride_qn,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_dob,
    stride_doh,
    stride_don,
    stride_dod,
    stride_ob,
    stride_oh,
    stride_on,
    stride_od,
    stride_dqb,
    stride_dqh,
    stride_dqn,
    stride_dqd,
    stride_lseb,
    stride_lseh,
    stride_lsen,
    stride_db,
    stride_dh,
    stride_dn,
    N_Q_HEADS: tl.constexpr,
    N_KV_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    scale,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SLIDE_SIZE: tl.constexpr,
):
    q_block_idx = tl.program_id(0)
    bh_idx = tl.program_id(1)
    q_h_idx = bh_idx % N_Q_HEADS
    b_idx = bh_idx // N_Q_HEADS
    kv_h_idx = q_h_idx * N_KV_HEADS // N_Q_HEADS

    q_offsets = q_block_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = q_offsets < SEQ_LEN
    d = tl.arange(0, HEAD_DIM)

    q_base = Q + b_idx * stride_qb + q_h_idx * stride_qh
    k_base = K + b_idx * stride_kb + kv_h_idx * stride_kh
    v_base = V + b_idx * stride_vb + kv_h_idx * stride_vh
    do_base = dO + b_idx * stride_dob + q_h_idx * stride_doh
    dq_base = dQ + b_idx * stride_dqb + q_h_idx * stride_dqh

    q = tl.load(q_base + q_offsets[:, None] * stride_qn + d[None, :] * stride_qd, mask=q_mask[:, None], other=0.0)
    do = tl.load(do_base + q_offsets[:, None] * stride_don + d[None, :] * stride_dod, mask=q_mask[:, None], other=0.0)

    lse = tl.load(
        LSE + b_idx * stride_lseb + q_h_idx * stride_lseh + q_offsets * stride_lsen,
        mask=q_mask,
        other=0.0,
    )
    delta = tl.load(
        Delta + b_idx * stride_db + q_h_idx * stride_dh + q_offsets * stride_dn,
        mask=q_mask,
        other=0.0,
    )

    if IS_CAUSAL:
        kv_end = (q_block_idx + 1) * BLOCK_Q
    else:
        kv_end = SEQ_LEN
    if IS_CAUSAL and SLIDE_SIZE > 0:
        kv_min = tl.maximum(0, q_block_idx * BLOCK_Q - SLIDE_SIZE + 1)
        kv_loop_start = (kv_min // BLOCK_KV) * BLOCK_KV
    else:
        kv_loop_start = 0

    dq = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)
    log2e: tl.constexpr = 1.4426950408889634
    scale_log2e = scale * log2e
    lse_log2 = lse * log2e

    for kv_start in range(kv_loop_start, kv_end, BLOCK_KV):
        kv_offsets = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask = kv_offsets < SEQ_LEN
        k = tl.load(k_base + kv_offsets[:, None] * stride_kn + d[None, :] * stride_kd, mask=kv_mask[:, None], other=0.0)
        v = tl.load(v_base + kv_offsets[:, None] * stride_vn + d[None, :] * stride_vd, mask=kv_mask[:, None], other=0.0)
        scores = tl.dot(q.to(tl.float32), tl.trans(k.to(tl.float32)), input_precision="ieee").to(tl.float32) * scale_log2e
        if IS_CAUSAL:
            if SLIDE_SIZE > 0:
                valid = (kv_offsets[None, :] <= q_offsets[:, None]) & (
                    q_offsets[:, None] - kv_offsets[None, :] < SLIDE_SIZE
                ) & kv_mask[None, :]
            else:
                valid = (kv_offsets[None, :] <= q_offsets[:, None]) & kv_mask[None, :]
        else:
            valid = kv_mask[None, :]
        scores = tl.where(valid, scores, -float("inf"))
        p = tl.math.exp2(scores - lse_log2[:, None])
        dp = tl.dot(do.to(tl.float32), tl.trans(v.to(tl.float32)), input_precision="ieee").to(tl.float32)
        ds = tl.where(valid, p * (dp - delta[:, None]), 0.0)
        dq += tl.dot(ds, k.to(tl.float32), input_precision="ieee").to(tl.float32)

    dq *= scale
    dq = tl.where(q_offsets[:, None] == 0, 0.0, dq)
    tl.store(dq_base + q_offsets[:, None] * stride_dqn + d[None, :] * stride_dqd, dq.to(q.dtype), mask=q_mask[:, None])


@triton.jit
def _sm120_gemma_bwd_dkv_kernel(
    Q,
    K,
    V,
    dO,
    dK,
    dV,
    LSE,
    Delta,
    stride_qb,
    stride_qh,
    stride_qn,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_dob,
    stride_doh,
    stride_don,
    stride_dod,
    stride_dkb,
    stride_dkh,
    stride_dkn,
    stride_dkd,
    stride_dvb,
    stride_dvh,
    stride_dvn,
    stride_dvd,
    stride_lseb,
    stride_lseh,
    stride_lsen,
    stride_db,
    stride_dh,
    stride_dn,
    N_Q_HEADS: tl.constexpr,
    N_KV_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    scale,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SLIDE_SIZE: tl.constexpr,
):
    kv_block_idx = tl.program_id(0)
    bkvh_idx = tl.program_id(1)
    kv_h_idx = bkvh_idx % N_KV_HEADS
    b_idx = bkvh_idx // N_KV_HEADS

    kv_offsets = kv_block_idx * BLOCK_KV + tl.arange(0, BLOCK_KV)
    kv_mask = kv_offsets < SEQ_LEN
    d = tl.arange(0, HEAD_DIM)

    k_base = K + b_idx * stride_kb + kv_h_idx * stride_kh
    v_base = V + b_idx * stride_vb + kv_h_idx * stride_vh
    dk_base = dK + b_idx * stride_dkb + kv_h_idx * stride_dkh
    dv_base = dV + b_idx * stride_dvb + kv_h_idx * stride_dvh

    k = tl.load(k_base + kv_offsets[:, None] * stride_kn + d[None, :] * stride_kd, mask=kv_mask[:, None], other=0.0)
    v = tl.load(v_base + kv_offsets[:, None] * stride_vn + d[None, :] * stride_vd, mask=kv_mask[:, None], other=0.0)
    dk = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)

    if IS_CAUSAL:
        q_start = (kv_block_idx * BLOCK_KV // BLOCK_Q) * BLOCK_Q
    else:
        q_start = 0
    if IS_CAUSAL and SLIDE_SIZE > 0:
        kv_last = kv_block_idx * BLOCK_KV + BLOCK_KV - 1
        q_max = kv_last + SLIDE_SIZE - 1
        q_loop_end = tl.minimum(SEQ_LEN, ((q_max // BLOCK_Q) + 1) * BLOCK_Q)
    else:
        q_loop_end = SEQ_LEN

    log2e: tl.constexpr = 1.4426950408889634
    scale_log2e = scale * log2e

    for qh_offset in tl.static_range(GQA_RATIO):
        q_h_idx = kv_h_idx * GQA_RATIO + qh_offset
        q_base = Q + b_idx * stride_qb + q_h_idx * stride_qh
        do_base = dO + b_idx * stride_dob + q_h_idx * stride_doh
        lse_base = LSE + b_idx * stride_lseb + q_h_idx * stride_lseh
        delta_base = Delta + b_idx * stride_db + q_h_idx * stride_dh

        for q_pos in range(q_start, q_loop_end, BLOCK_Q):
            q_offsets = q_pos + tl.arange(0, BLOCK_Q)
            q_mask = q_offsets < SEQ_LEN
            q = tl.load(q_base + q_offsets[:, None] * stride_qn + d[None, :] * stride_qd, mask=q_mask[:, None], other=0.0)
            do = tl.load(do_base + q_offsets[:, None] * stride_don + d[None, :] * stride_dod, mask=q_mask[:, None], other=0.0)
            lse = tl.load(lse_base + q_offsets * stride_lsen, mask=q_mask, other=0.0)
            delta = tl.load(delta_base + q_offsets * stride_dn, mask=q_mask, other=0.0)
            scores = tl.dot(q, tl.trans(k)).to(tl.float32) * scale_log2e
            if IS_CAUSAL:
                if SLIDE_SIZE > 0:
                    valid = (kv_offsets[None, :] <= q_offsets[:, None]) & kv_mask[None, :] & q_mask[:, None] & (
                        q_offsets[:, None] - kv_offsets[None, :] < SLIDE_SIZE
                    )
                else:
                    valid = (kv_offsets[None, :] <= q_offsets[:, None]) & kv_mask[None, :] & q_mask[:, None]
            else:
                valid = kv_mask[None, :] & q_mask[:, None]
            scores = tl.where(valid, scores, -float("inf"))
            p = tl.math.exp2(scores - (lse * log2e)[:, None])
            if HEAD_DIM >= 512:
                dv += tl.dot(tl.trans(p), do.to(tl.float32), input_precision="ieee").to(tl.float32)
                dp = tl.dot(do.to(tl.float32), tl.trans(v.to(tl.float32)), input_precision="ieee").to(tl.float32)
            else:
                dv += tl.dot(tl.trans(p.to(do.dtype)), do).to(tl.float32)
                dp = tl.dot(do, tl.trans(v)).to(tl.float32)
            ds = tl.where(valid, p * (dp - delta[:, None]), 0.0)
            if HEAD_DIM >= 512:
                dk += tl.dot(tl.trans(ds), q.to(tl.float32), input_precision="ieee").to(tl.float32)
            else:
                dk += tl.dot(tl.trans(ds.to(q.dtype)), q).to(tl.float32)

    dk *= scale
    tl.store(dk_base + kv_offsets[:, None] * stride_dkn + d[None, :] * stride_dkd, dk.to(k.dtype), mask=kv_mask[:, None])
    tl.store(dv_base + kv_offsets[:, None] * stride_dvn + d[None, :] * stride_dvd, dv.to(v.dtype), mask=kv_mask[:, None])


def _triton_dense_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    *,
    causal: bool,
    window_size_left: Optional[int],
    window_size_right: Optional[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, h_q, seqlen, head_dim = q.shape
    h_kv = k.shape[1]
    scale = _default_scale(head_dim)
    slide_size = _slide_size(window_size_left, window_size_right)
    if slide_size >= seqlen:
        slide_size = 0
    if head_dim >= 512:
        block_q_dq, block_kv_dq = 8, 16
        block_kv_dkv, block_q_dkv = 8, 16
        num_warps = 8
    else:
        block_q_dq, block_kv_dq = 16, 16
        block_kv_dkv, block_q_dkv = 64, 16
        num_warps = 8
    block_q_dq = min(block_q_dq, triton.next_power_of_2(seqlen))
    block_kv_dq = max(16, min(block_kv_dq, triton.next_power_of_2(seqlen)))
    block_kv_dkv = min(block_kv_dkv, triton.next_power_of_2(seqlen))
    block_q_dkv = max(16, min(block_q_dkv, triton.next_power_of_2(seqlen)))

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    delta = torch.empty((batch, h_q, seqlen), dtype=torch.float32, device=q.device)

    _sm120_gemma_delta_kernel[(triton.cdiv(seqlen, block_q_dq), batch * h_q)](
        q,
        k,
        v,
        dout,
        lse,
        delta,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        dout.stride(0),
        dout.stride(1),
        dout.stride(2),
        dout.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        delta.stride(0),
        delta.stride(1),
        delta.stride(2),
        N_Q_HEADS=h_q,
        N_KV_HEADS=h_kv,
        SEQ_LEN=seqlen,
        HEAD_DIM=head_dim,
        scale=scale,
        BLOCK_Q=block_q_dq,
        BLOCK_KV=block_kv_dq,
        IS_CAUSAL=causal,
        SLIDE_SIZE=slide_size,
        num_warps=num_warps,
        num_stages=1,
    )
    _sm120_gemma_bwd_dkv_kernel[(triton.cdiv(seqlen, block_kv_dkv), batch * h_kv)](
        q,
        k,
        v,
        dout,
        dk,
        dv,
        lse,
        delta,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        dout.stride(0),
        dout.stride(1),
        dout.stride(2),
        dout.stride(3),
        dk.stride(0),
        dk.stride(1),
        dk.stride(2),
        dk.stride(3),
        dv.stride(0),
        dv.stride(1),
        dv.stride(2),
        dv.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        delta.stride(0),
        delta.stride(1),
        delta.stride(2),
        N_Q_HEADS=h_q,
        N_KV_HEADS=h_kv,
        SEQ_LEN=seqlen,
        HEAD_DIM=head_dim,
        scale=scale,
        BLOCK_Q=block_q_dkv,
        BLOCK_KV=block_kv_dkv,
        GQA_RATIO=h_q // h_kv,
        IS_CAUSAL=causal,
        SLIDE_SIZE=slide_size,
        num_warps=num_warps,
        num_stages=1,
    )
    _sm120_gemma_bwd_dq_kernel[(triton.cdiv(seqlen, block_q_dq), batch * h_q)](
        q,
        k,
        v,
        dout,
        out,
        dq,
        lse,
        delta,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        dout.stride(0),
        dout.stride(1),
        dout.stride(2),
        dout.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        dq.stride(0),
        dq.stride(1),
        dq.stride(2),
        dq.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        delta.stride(0),
        delta.stride(1),
        delta.stride(2),
        N_Q_HEADS=h_q,
        N_KV_HEADS=h_kv,
        SEQ_LEN=seqlen,
        HEAD_DIM=head_dim,
        scale=scale,
        BLOCK_Q=block_q_dq,
        BLOCK_KV=block_kv_dq,
        IS_CAUSAL=causal,
        SLIDE_SIZE=slide_size,
        num_warps=num_warps,
        num_stages=1,
    )
    return dq, dk, dv


def supports_dense(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: Optional[float],
    causal: bool,
    window_size_left: Optional[int],
    window_size_right: Optional[int],
) -> bool:
    if _device_arch(q) // 10 != 12 or q.dtype not in (torch.float16, torch.bfloat16):
        return False
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return False
    if not causal or window_size_right not in (None, 0):
        return False
    if q.shape[0] != k.shape[0] or q.shape[1] != k.shape[1] or k.shape != v.shape:
        return False
    _, _, h_q, d = q.shape
    h_kv = k.shape[2]
    if not _is_default_scale(softmax_scale, d):
        return False
    return (h_q, h_kv, d) in ((32, 16, 256), (32, 4, 512))


def supports_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: Optional[float],
    causal: bool,
    window_size_left: Optional[int],
    window_size_right: Optional[int],
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
) -> bool:
    if _device_arch(q) // 10 != 12 or q.dtype not in (torch.float16, torch.bfloat16):
        return False
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        return False
    if cu_seqlens_q is None or cu_seqlens_k is None:
        return False
    if not causal or window_size_right not in (None, 0):
        return False
    if k.shape != v.shape:
        return False
    _, h_q, d = q.shape
    h_kv = k.shape[1]
    if not _is_default_scale(softmax_scale, d):
        return False
    return (h_q, h_kv, d) == (32, 16, 256)


class _Sm120GemmaDenseFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale: Optional[float],
        causal: bool,
        window_size_left: Optional[int],
        window_size_right: Optional[int],
    ):
        from flash_attn.cute.interface import _flash_attn_fwd

        out, lse = _flash_attn_fwd(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            return_lse=True,
            num_splits=1,
            pack_gqa=False,
        )
        if q.shape[-1] == 512:
            lse = _triton_dense_lse(
                q,
                k,
                causal=causal,
                window_size_left=window_size_left,
                window_size_right=window_size_right,
            )
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.causal = causal
        ctx.window_size_left = window_size_left
        ctx.window_size_right = window_size_right
        return out, lse

    @staticmethod
    def backward(ctx, dout, dlse):
        del dlse
        q, k, v, out, lse = ctx.saved_tensors
        if dout is None:
            dout = torch.zeros_like(q)

        q_t = q.detach().transpose(1, 2).contiguous()
        k_t = k.detach().transpose(1, 2).contiguous()
        v_t = v.detach().transpose(1, 2).contiguous()
        out_t = out.detach().transpose(1, 2).contiguous()
        dout_t = dout.detach().transpose(1, 2).contiguous()
        dq_t, dk_t, dv_t = _triton_dense_backward(
            q_t,
            k_t,
            v_t,
            out_t,
            dout_t,
            lse.detach(),
            causal=ctx.causal,
            window_size_left=ctx.window_size_left,
            window_size_right=ctx.window_size_right,
        )
        return (
            dq_t.transpose(1, 2).contiguous(),
            dk_t.transpose(1, 2).contiguous(),
            dv_t.transpose(1, 2).contiguous(),
            None,
            None,
            None,
            None,
        )


class _Sm120GemmaVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: Optional[float],
        causal: bool,
        window_size_left: Optional[int],
        window_size_right: Optional[int],
    ):
        from flash_attn.cute.interface import _flash_attn_fwd

        cu_q_cpu = cu_seqlens_q.detach().cpu()
        cu_k_cpu = cu_seqlens_k.detach().cpu()
        has_empty_sequence = bool(
            ((cu_q_cpu[1:] - cu_q_cpu[:-1]) == 0).any()
            or ((cu_k_cpu[1:] - cu_k_cpu[:-1]) == 0).any()
        )
        if has_empty_sequence:
            out = torch.empty_like(q)
            lse = torch.empty((q.shape[1], q.shape[0]), device=q.device, dtype=torch.float32)
            for b in range(cu_q_cpu.numel() - 1):
                q_start, q_end = int(cu_q_cpu[b]), int(cu_q_cpu[b + 1])
                k_start, k_end = int(cu_k_cpu[b]), int(cu_k_cpu[b + 1])
                if q_end == q_start:
                    continue
                if k_end == k_start:
                    out[q_start:q_end].zero_()
                    lse[:, q_start:q_end].fill_(-torch.inf)
                    continue
                out_b, lse_b = _flash_attn_fwd(
                    q[q_start:q_end].unsqueeze(0),
                    k[k_start:k_end].unsqueeze(0),
                    v[k_start:k_end].unsqueeze(0),
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size_left=window_size_left,
                    window_size_right=window_size_right,
                    return_lse=True,
                    num_splits=1,
                    pack_gqa=False,
                )
                out[q_start:q_end] = out_b.squeeze(0)
                lse[:, q_start:q_end] = lse_b.squeeze(0)
        else:
            out, lse = _flash_attn_fwd(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size_left=window_size_left,
                window_size_right=window_size_right,
                return_lse=True,
                num_splits=1,
                pack_gqa=False,
            )
        ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k)
        ctx.causal = causal
        ctx.window_size_left = window_size_left
        ctx.window_size_right = window_size_right
        return out, lse

    @staticmethod
    def backward(ctx, dout, dlse):
        del dlse
        q, k, v, out, lse, cu_q, cu_k = ctx.saved_tensors
        if dout is None:
            dout = torch.zeros_like(q)

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        cu_q_cpu = cu_q.detach().cpu()
        cu_k_cpu = cu_k.detach().cpu()

        for b in range(cu_q_cpu.numel() - 1):
            q_start, q_end = int(cu_q_cpu[b]), int(cu_q_cpu[b + 1])
            k_start, k_end = int(cu_k_cpu[b]), int(cu_k_cpu[b + 1])
            if q_end == q_start:
                continue
            if k_end == k_start:
                dq[q_start:q_end].zero_()
                continue
            if q_end - q_start != k_end - k_start:
                raise NotImplementedError("SM120 varlen backward currently requires self-attention packing")
            q_t = q[q_start:q_end].detach().transpose(0, 1).unsqueeze(0).contiguous()
            k_t = k[k_start:k_end].detach().transpose(0, 1).unsqueeze(0).contiguous()
            v_t = v[k_start:k_end].detach().transpose(0, 1).unsqueeze(0).contiguous()
            out_t = out[q_start:q_end].detach().transpose(0, 1).unsqueeze(0).contiguous()
            dout_t = dout[q_start:q_end].detach().transpose(0, 1).unsqueeze(0).contiguous()
            lse_t = lse[:, q_start:q_end].detach().unsqueeze(0).contiguous()
            dq_t, dk_t, dv_t = _triton_dense_backward(
                q_t,
                k_t,
                v_t,
                out_t,
                dout_t,
                lse_t,
                causal=ctx.causal,
                window_size_left=ctx.window_size_left,
                window_size_right=ctx.window_size_right,
            )
            dq[q_start:q_end] = dq_t.squeeze(0).transpose(0, 1)
            dk[k_start:k_end] = dk_t.squeeze(0).transpose(0, 1)
            dv[k_start:k_end] = dv_t.squeeze(0).transpose(0, 1)

        return dq, dk, dv, None, None, None, None, None, None, None, None


def flash_attn_sm120_gemma_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: Optional[float],
    causal: bool,
    window_size_left: Optional[int],
    window_size_right: Optional[int],
):
    return _Sm120GemmaDenseFunc.apply(
        q, k, v, softmax_scale, causal, window_size_left, window_size_right
    )


def flash_attn_sm120_gemma_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: Optional[float],
    causal: bool,
    window_size_left: Optional[int],
    window_size_right: Optional[int],
):
    return _Sm120GemmaVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
    )
