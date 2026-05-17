import math
from functools import partial
from typing import Callable, Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, warp
import torch

from quack import layout_utils

from flash_attn.cute import ampere_helpers as sm80_utils
from flash_attn.cute import utils
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute.cache_utils import get_jit_cache
from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned, to_cute_tensor, torch2cute_dtype_map
from flash_attn.cute.flash_fwd import FlashAttentionForwardSm80
from flash_attn.cute.mask import AttentionMask
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.tile_scheduler import SingleTileScheduler, TileSchedulerArguments


class Sm120GemmaBackwardDQ(FlashAttentionForwardSm80):
    """SM120 Gemma split-backward Pass B: one CTA computes one Q tile's dQ."""

    def _setup_attributes(self):
        super()._setup_attributes()
        self.sLSE_layout = cute.make_layout((self.tile_m,), stride=(1,))
        self.sLSEMma_layout = cute.make_layout((self.tile_m, self.tile_n), stride=(1, 0))

        universal_copy_bits = 128
        async_copy_elems_accum = universal_copy_bits // Float32.width
        atom_async_copy_accum = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            Float32,
            num_bits_per_copy=universal_copy_bits,
        )
        self.gmem_tiled_copy_LSE = cute.make_tiled_copy_tv(
            atom_async_copy_accum,
            cute.make_layout(self.num_threads),
            cute.make_layout(async_copy_elems_accum),
        )

    def _get_shared_storage_cls(self):
        sQ_struct, sK_struct, sV_struct = [
            cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(layout)], 1024]
            for layout in (self.sQ_layout, self.sK_layout, self.sV_layout)
        ]
        sLSE_struct, sdPsum_struct = [
            cute.struct.Align[cute.struct.MemRange[Float32, cute.cosize(layout)], 128]
            for layout in (self.sLSE_layout, self.sLSE_layout)
        ]

        @cute.struct
        class SharedStorage:
            sQ: sQ_struct
            sK: sK_struct
            sV: sV_struct
            sLSE: sLSE_struct
            sdPsum: sdPsum_struct

        return SharedStorage

    @cute.jit
    def _load_rowsum(
        self,
        gmem_tiled_copy: cute.TiledCopy,
        tG: cute.Tensor,
        tS: cute.Tensor,
        tC: cute.Tensor,
    ):
        for m in cutlass.range_constexpr(cute.size(tS.shape[1])):
            if tC[0, m][0] < self.tile_m:
                cute.copy(gmem_tiled_copy, tG[None, m], tS[None, m])

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSElog2: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQ: cute.Tensor,
        softmax_scale: Float32,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
        stream: cuda.CUstream = None,
    ):
        self._check_type(
            *(
                t.element_type if t is not None else None
                for t in (mQ, mK, mV, mdQ, None, None, None, None, None)
            )
        )
        if const_expr(not (mdO.element_type == self.dtype)):
            raise TypeError("dO tensor must have the same dtype as Q")
        if const_expr(not (mLSElog2.element_type == mdPsum.element_type == Float32)):
            raise TypeError("LSE/log2 and dPsum tensors must be Float32")

        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
        self.num_mma_threads = tiled_mma_pv.size
        self.num_producer_threads = self.num_threads
        self.num_Q_load_threads = self.num_threads
        self.num_epilogue_threads = self.num_threads
        self.use_tma_O = False
        self._setup_attributes()
        SharedStorage = self._get_shared_storage_cls()

        mQ, mK, mV, mdO, mdQ, mLSElog2, mdPsum = [
            assume_tensor_aligned(t) for t in (mQ, mK, mV, mdO, mdQ, mLSElog2, mdPsum)
        ]
        q_layout_transpose = [1, 3, 2, 0]
        kv_layout_transpose = [1, 3, 2, 0]
        rowsum_layout_transpose = [2, 1, 0]
        mQ, mdO, mdQ = [
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=q_layout_transpose))
            for t in (mQ, mdO, mdQ)
        ]
        mK, mV = [
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=kv_layout_transpose))
            for t in (mK, mV)
        ]
        mLSElog2, mdPsum = [
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=rowsum_layout_transpose))
            for t in (mLSElog2, mdPsum)
        ]

        tile_sched_args = TileSchedulerArguments(
            num_block=cute.ceil_div(mQ.shape[0], self.tile_m),
            num_head=cute.size(mQ.shape[2]),
            num_batch=mQ.shape[3],
            num_splits=1,
            seqlen_k=0,
            headdim=mQ.shape[1],
            headdim_v=mV.shape[1],
            total_q=cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=(self.tile_m, self.tile_n),
            qhead_per_kvhead_packgqa=1,
        )
        tile_sched_params = SingleTileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = SingleTileScheduler.get_grid_shape(tile_sched_params)
        softmax_scale_log2 = softmax_scale * math.log2(math.e)

        self.kernel(
            mQ,
            mK,
            mV,
            mdO,
            mLSElog2,
            mdPsum,
            mdQ,
            softmax_scale,
            softmax_scale_log2,
            window_size_left,
            window_size_right,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sO_layout,
            self.sLSE_layout,
            self.sLSEMma_layout,
            self.gmem_tiled_copy_Q,
            self.gmem_tiled_copy_K,
            self.gmem_tiled_copy_V,
            self.gmem_tiled_copy_O,
            self.gmem_tiled_copy_LSE,
            tiled_mma_qk,
            tiled_mma_pv,
            SharedStorage,
            tile_sched_params,
            SingleTileScheduler,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSElog2: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQ: cute.Tensor,
        softmax_scale: Float32,
        softmax_scale_log2: Float32,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        sLSE_layout: cute.Layout,
        sLSEMma_layout: cute.Layout,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_K: cute.TiledCopy,
        gmem_tiled_copy_V: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        gmem_tiled_copy_LSE: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
        tile_sched_params,
        TileScheduler: cutlass.Constexpr[Callable],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        tile_scheduler = TileScheduler.create(tile_sched_params)
        work_tile = tile_scheduler.initial_work_tile_info()
        m_block, head_idx, batch_idx, _ = work_tile.tile_idx

        if work_tile.is_valid_tile:
            block_info = BlockInfo(
                self.tile_m,
                self.tile_n,
                self.is_causal,
                self.is_local,
                False,
                window_size_left,
                window_size_right,
            )
            seqlen = SeqlenInfoQK.create(
                batch_idx=batch_idx,
                seqlen_q_static=mQ.shape[0],
                seqlen_k_static=mK.shape[0],
                tile_m=self.tile_m,
                tile_n=self.tile_n,
            )
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)

            blkQ_shape = (self.tile_m, self.tile_hdim)
            blkK_shape = (self.tile_n, self.tile_hdim)
            blkV_shape = (self.tile_n, self.tile_hdimv)
            head_idx_kv = head_idx // self.qhead_per_kvhead
            mQ_cur = mQ[None, None, head_idx, batch_idx]
            mdO_cur = mdO[None, None, head_idx, batch_idx]
            mdQ_cur = mdQ[None, None, head_idx, batch_idx]
            mK_cur = mK[None, None, head_idx_kv, batch_idx]
            mV_cur = mV[None, None, head_idx_kv, batch_idx]
            mLSE_cur = mLSElog2[None, head_idx, batch_idx]
            mdPsum_cur = mdPsum[None, head_idx, batch_idx]

            gQ = cute.local_tile(mQ_cur, blkQ_shape, (m_block, 0))
            gdO = cute.local_tile(mdO_cur, blkQ_shape, (m_block, 0))
            gK = cute.local_tile(mK_cur, blkK_shape, (None, 0))
            gV = cute.local_tile(mV_cur, blkV_shape, (None, 0))
            gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (m_block,))
            gdPsum = cute.local_tile(mdPsum_cur, (self.tile_m,), (m_block,))

            smem = cutlass.utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)
            sQ = storage.sQ.get_tensor(sQ_layout)
            sK = storage.sK.get_tensor(sK_layout)
            sV = storage.sV.get_tensor(sV_layout)
            sVt = layout_utils.transpose_view(sV)
            sKt = layout_utils.transpose_view(sK)
            sLSE = storage.sLSE.get_tensor(sLSE_layout)
            sdPsum = storage.sdPsum.get_tensor(sLSE_layout)
            sLSEMma = storage.sLSE.get_tensor(sLSEMma_layout)
            sdPsumMma = storage.sdPsum.get_tensor(sLSEMma_layout)

            gmem_thr_copy_Q = gmem_tiled_copy_Q.get_slice(tidx)
            gmem_thr_copy_K = gmem_tiled_copy_K.get_slice(tidx)
            gmem_thr_copy_V = gmem_tiled_copy_V.get_slice(tidx)
            gmem_thr_copy_LSE = gmem_tiled_copy_LSE.get_slice(tidx)

            tKgK = gmem_thr_copy_K.partition_S(gK)
            tKsK = gmem_thr_copy_K.partition_D(sK)
            tVgV = gmem_thr_copy_V.partition_S(gV)
            tVsV = gmem_thr_copy_V.partition_D(sV)
            cK = cute.make_identity_tensor((self.tile_n, self.tile_hdim))
            tKcK = gmem_thr_copy_K.partition_S(cK)
            t0KcK = gmem_tiled_copy_K.get_slice(0).partition_S(cK)
            tKpK = utils.predicate_k(tKcK, limit=mK.shape[1])
            cV = cute.make_identity_tensor((self.tile_n, self.tile_hdimv))
            tVcV = gmem_thr_copy_V.partition_S(cV)
            t0VcV = gmem_tiled_copy_V.get_slice(0).partition_S(cV)
            tVpV = utils.predicate_k(tVcV, limit=mV.shape[1])
            tLSEgLSE = gmem_thr_copy_LSE.partition_S(gLSE)
            tLSEsLSE = gmem_thr_copy_LSE.partition_D(sLSE)
            tdPsumgdPsum = gmem_thr_copy_LSE.partition_S(gdPsum)
            tdPsumsdPsum = gmem_thr_copy_LSE.partition_D(sdPsum)
            cLSE = cute.make_identity_tensor((self.tile_m,))
            tLSEcLSE = gmem_thr_copy_LSE.partition_S(cLSE)

            thr_mma_qk = tiled_mma_qk.get_slice(tidx)
            thr_mma_pv = tiled_mma_pv.get_slice(tidx)
            tSrQ = thr_mma_qk.make_fragment_A(thr_mma_qk.partition_A(sQ))
            tdPrdO = thr_mma_qk.make_fragment_A(thr_mma_qk.partition_A(sQ))
            tSrK = thr_mma_qk.make_fragment_B(thr_mma_qk.partition_B(sK[None, None, 0]))
            tdPrV = thr_mma_qk.make_fragment_B(thr_mma_qk.partition_B(sV[None, None, 0]))
            tdQrKt = thr_mma_pv.make_fragment_B(thr_mma_pv.partition_B(sKt[None, None, 0]))
            acc_shape_dQ = thr_mma_pv.partition_shape_C((self.tile_m, self.tile_hdim))
            acc_dQ = cute.make_fragment(acc_shape_dQ, Float32)
            acc_dQ.fill(0.0)

            smem_copy_atom = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                self.dtype,
            )
            smem_copy_atom_transposed = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
                self.dtype,
            )
            smem_thr_copy_Q = utils.make_tiled_copy_A(smem_copy_atom, tiled_mma_qk).get_slice(tidx)
            smem_thr_copy_K = utils.make_tiled_copy_B(smem_copy_atom, tiled_mma_qk).get_slice(tidx)
            smem_thr_copy_V = utils.make_tiled_copy_B(smem_copy_atom, tiled_mma_qk).get_slice(tidx)
            smem_thr_copy_Kt = utils.make_tiled_copy_B(smem_copy_atom_transposed, tiled_mma_pv).get_slice(tidx)
            tSsQ = smem_thr_copy_Q.partition_S(sQ)
            tSsK = smem_thr_copy_K.partition_S(sK)
            tdPsV = smem_thr_copy_V.partition_S(sV)
            tdQsKt = smem_thr_copy_Kt.partition_S(sKt)
            tSsLSEMma = layout_utils.reshape_acc_to_mn(thr_mma_qk.partition_C(sLSEMma))[None, 0]
            tSsdPsumMma = layout_utils.reshape_acc_to_mn(thr_mma_qk.partition_C(sdPsumMma))[None, 0]

            self.load_Q(gmem_thr_copy_Q, gQ, sQ, m_block, seqlen=seqlen.seqlen_q, headdim=mQ.shape[1])
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()
            tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
            cute.copy(smem_thr_copy_Q, tSsQ, tSrQ_copy_view)
            cute.arch.barrier()

            self.load_Q(gmem_thr_copy_Q, gdO, sQ, m_block, seqlen=seqlen.seqlen_q, headdim=mdO.shape[1])
            cute.arch.cp_async_commit_group()
            self._load_rowsum(gmem_tiled_copy_LSE, tLSEgLSE, tLSEsLSE, tLSEcLSE)
            cute.arch.cp_async_commit_group()
            self._load_rowsum(gmem_tiled_copy_LSE, tdPsumgdPsum, tdPsumsdPsum, tLSEcLSE)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()
            tdPrdO_copy_view = smem_thr_copy_Q.retile(tdPrdO)
            cute.copy(smem_thr_copy_Q, tSsQ, tdPrdO_copy_view)

            tLSErLSE = cute.make_fragment_like(tSsLSEMma)
            cute.autovec_copy(tSsLSEMma, tLSErLSE)
            tLSErdPsum = cute.make_fragment_like(tSsdPsumMma)
            cute.autovec_copy(tSsdPsumMma, tLSErdPsum)

            mask = AttentionMask(self.tile_m, self.tile_n, seqlen, window_size_left, window_size_right)
            mask_fn = partial(
                mask.apply_mask,
                batch_idx=batch_idx,
                head_idx=head_idx,
                m_block=m_block,
                thr_mma=thr_mma_qk,
                mask_causal=self.is_causal and not self.is_local,
                mask_local=self.is_local,
            )

            for n_block in cutlass.range(n_block_min, n_block_max, unroll=1):
                self.load_K(
                    gmem_tiled_copy_K,
                    tKgK,
                    tKsK,
                    tKcK,
                    t0KcK,
                    tKpK,
                    n_block,
                    0,
                    seqlen.seqlen_k,
                    True,
                )
                cute.arch.cp_async_commit_group()
                self.load_V(
                    gmem_tiled_copy_V,
                    tVgV,
                    tVsV,
                    tVcV,
                    t0VcV,
                    tVpV,
                    n_block,
                    0,
                    seqlen.seqlen_k,
                    True,
                )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.barrier()

                acc_shape_S = thr_mma_qk.partition_shape_C((self.tile_m, self.tile_n))
                acc_S = cute.make_fragment(acc_shape_S, Float32)
                acc_S.fill(0.0)
                sm80_utils.gemm(
                    tiled_mma_qk,
                    acc_S,
                    tSrQ,
                    tSrK,
                    tSsQ,
                    tSsK[None, None, None, 0],
                    smem_thr_copy_Q,
                    smem_thr_copy_K,
                    A_in_regs=True,
                )
                mask_fn(acc_S, n_block=n_block, mask_seqlen=True)
                acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
                for r in cutlass.range(cute.size(acc_S_mn, mode=[0]), unroll_full=True):
                    acc_S_mn[r, None].store(
                        cute.math.exp2(
                            acc_S_mn[r, None].load() * softmax_scale_log2 - tLSErLSE[r],
                            fastmath=True,
                        )
                    )

                acc_dP = cute.make_fragment(acc_shape_S, Float32)
                acc_dP.fill(0.0)
                sm80_utils.gemm(
                    tiled_mma_qk,
                    acc_dP,
                    tdPrdO,
                    tdPrV,
                    tSsQ,
                    tdPsV[None, None, None, 0],
                    smem_thr_copy_Q,
                    smem_thr_copy_V,
                    A_in_regs=True,
                )
                acc_dP_mn = layout_utils.reshape_acc_to_mn(acc_dP)
                for r in cutlass.range(cute.size(acc_dP_mn, mode=[0]), unroll_full=True):
                    acc_dP_mn[r, None].store(
                        acc_S_mn[r, None].load() * (acc_dP_mn[r, None].load() - tLSErdPsum[r])
                    )
                rdS = cute.make_fragment_like(acc_dP, self.dtype)
                rdS.store(acc_dP.load().to(self.dtype))
                tdQrdS = layout_utils.reshape_acc_to_frgA(rdS)
                sm80_utils.gemm_rs(
                    tiled_mma_pv,
                    acc_dQ,
                    tdQrdS,
                    tdQrKt,
                    tdQsKt[None, None, None, 0],
                    smem_thr_copy_Kt,
                )

            acc_dQ.store(acc_dQ.load() * softmax_scale)
            cDQ = cute.make_identity_tensor((self.tile_m, self.tile_hdim))
            tDQcDQ = layout_utils.reshape_acc_to_mn(thr_mma_pv.partition_C(cDQ))
            acc_dQ_mn = layout_utils.reshape_acc_to_mn(acc_dQ)
            for r in cutlass.range(cute.size(acc_dQ_mn, mode=[0]), unroll_full=True):
                if tDQcDQ[r, 0][0] + m_block * self.tile_m == 0:
                    acc_dQ_mn[r, None].store(acc_dQ_mn[r, None].load() * 0.0)
            sO = cute.make_tensor(sQ.iterator, sO_layout)
            self.epilogue(
                acc_dQ,
                tLSErLSE,
                mdQ,
                None,
                sO,
                seqlen,
                gmem_tiled_copy_O,
                None,
                tiled_mma_pv,
                tidx,
                m_block,
                head_idx,
                batch_idx,
            )


def _dq_config(head_dim: int) -> tuple[int, int, int]:
    if head_dim >= 512:
        return 32, 32, 128
    return 64, 64, 128


def sm120_gemma_cute_dq(
    q,
    k,
    v,
    out,
    dout,
    lse,
    *,
    causal: bool,
    window_size_left: Optional[int],
    window_size_right: Optional[int],
):
    batch, seqlen, h_q, head_dim = q.shape
    h_kv = k.shape[2]
    if head_dim >= 512:
        raise NotImplementedError("SM120 Gemma D512 dQ needs output-dimension chunking")
    tile_m, tile_n, num_threads = _dq_config(head_dim)
    seqlen_rounded = (seqlen + tile_m - 1) // tile_m * tile_m
    dpsum = q.new_empty((batch, h_q, seqlen_rounded), dtype=torch.float32)
    lse_log2 = q.new_empty((batch, h_q, seqlen_rounded), dtype=torch.float32)

    dtype = torch2cute_dtype_map[q.dtype]
    from flash_attn.cute.interface import _bwd_preprocess

    _bwd_preprocess(
        out,
        dout,
        dpsum,
        lse,
        lse_log2,
        None,
        None,
        None,
        None,
        dtype,
        head_dim,
        head_dim,
        tile_m,
    )
    dq = q.new_empty(q.shape)
    compile_key = (
        dtype,
        head_dim,
        h_q // h_kv,
        causal,
        window_size_left is not None,
        window_size_right is not None,
        tile_m,
        tile_n,
        num_threads,
    )
    if compile_key not in sm120_gemma_cute_dq.compile_cache:
        obj = Sm120GemmaBackwardDQ(
            dtype,
            head_dim,
            head_dim,
            qhead_per_kvhead=h_q // h_kv,
            is_causal=causal,
            is_local=window_size_left is not None or window_size_right is not None,
            pack_gqa=False,
            tile_m=tile_m,
            tile_n=tile_n,
            num_stages=1,
            num_threads=num_threads,
            Q_in_regs=True,
        )
        sm120_gemma_cute_dq.compile_cache[compile_key] = cute.compile(
            obj,
            to_cute_tensor(q),
            to_cute_tensor(k),
            to_cute_tensor(v),
            to_cute_tensor(dout),
            to_cute_tensor(lse_log2),
            to_cute_tensor(dpsum),
            to_cute_tensor(dq),
            Float32(1.0 / math.sqrt(head_dim)),
            Int32(0) if window_size_left is not None else None,
            Int32(0) if window_size_right is not None else None,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    sm120_gemma_cute_dq.compile_cache[compile_key](
        q.detach(),
        k.detach(),
        v.detach(),
        dout,
        lse_log2,
        dpsum,
        dq,
        1.0 / math.sqrt(head_dim),
        window_size_left,
        window_size_right,
    )
    return dq

sm120_gemma_cute_dq.compile_cache = get_jit_cache("sm120_gemma_dq")
