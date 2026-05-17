import math
from typing import Optional

import torch


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

        from flash_attn.cute.sm120_gemma_bwd import sm120_gemma_cute_dkv, sm120_gemma_cute_dq

        q_c = q.detach().contiguous()
        k_c = k.detach().contiguous()
        v_c = v.detach().contiguous()
        out_c = out.detach().contiguous()
        dout_c = dout.detach().contiguous()
        lse_c = lse.detach().contiguous()
        dq = sm120_gemma_cute_dq(
            q_c,
            k_c,
            v_c,
            out_c,
            dout_c,
            lse_c,
            causal=ctx.causal,
            window_size_left=ctx.window_size_left,
            window_size_right=ctx.window_size_right,
        )
        dk, dv = sm120_gemma_cute_dkv(
            q_c,
            k_c,
            v_c,
            out_c,
            dout_c,
            lse_c,
            causal=ctx.causal,
            window_size_left=ctx.window_size_left,
            window_size_right=ctx.window_size_right,
        )
        return (
            dq,
            dk,
            dv,
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

            from flash_attn.cute.sm120_gemma_bwd import sm120_gemma_cute_dkv, sm120_gemma_cute_dq

            q_b = q[q_start:q_end].detach().unsqueeze(0).contiguous()
            k_b = k[k_start:k_end].detach().unsqueeze(0).contiguous()
            v_b = v[k_start:k_end].detach().unsqueeze(0).contiguous()
            out_b = out[q_start:q_end].detach().unsqueeze(0).contiguous()
            dout_b = dout[q_start:q_end].detach().unsqueeze(0).contiguous()
            lse_b = lse[:, q_start:q_end].detach().unsqueeze(0).contiguous()
            dq_b = sm120_gemma_cute_dq(
                q_b,
                k_b,
                v_b,
                out_b,
                dout_b,
                lse_b,
                causal=ctx.causal,
                window_size_left=ctx.window_size_left,
                window_size_right=ctx.window_size_right,
            )
            dk_b, dv_b = sm120_gemma_cute_dkv(
                q_b,
                k_b,
                v_b,
                out_b,
                dout_b,
                lse_b,
                causal=ctx.causal,
                window_size_left=ctx.window_size_left,
                window_size_right=ctx.window_size_right,
            )
            dq[q_start:q_end] = dq_b.squeeze(0)
            dk[k_start:k_end] = dk_b.squeeze(0)
            dv[k_start:k_end] = dv_b.squeeze(0)

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
