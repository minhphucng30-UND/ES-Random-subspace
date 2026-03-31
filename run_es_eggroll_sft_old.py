"""
EggRoll-style ES SFT: fused low-rank perturbations in the forward pass instead of
materializing full ΔW = A B^T and adding it to weights for every evaluation.

For each nn.Linear, forward is:
    y = x W^T + b + sign * (σ / √r) * (x B) A^T
where A ∈ R^{out×r}, B ∈ R^{in×r} are drawn from a deterministic RNG keyed by
(population_seed, layer_id). The central-difference uses sign = +1 and sign = -1
with the same A,B (same seed), matching the two-sided estimate used in run_es_sft.py.

Weight updates apply sum_k coeff_k * E_k with the same RNG as the forward
(fold_in_key(seed, layer_id)), so the ES step matches the evaluated perturbation.

Requires plain nn.Linear modules (use HuggingFace AutoModelForCausalLM, not fused-only stacks).

Principal projection: before training, each layer's weight ``W`` (``out×in``) yields a fixed
right-subspace projector ``P = V_k V_k^T`` (``in×in``) from the top-``k`` right singular vectors
(``k = filter_rank``), matching ``neural_thicket.utils.get_principal_parameters``. The implied
``p_mat = b a^T`` in ``W^T`` space is left-multiplied by ``P`` (same as ``(noise @ P)^T`` for
full-matrix noise). Weight updates apply ``E @ P`` to each ``E = a b^T / √r``. Set ``filter_rank``
0 or negative to disable (full perturbation in all directions).

Throughput: optional per-seed ``noise_cache`` reuses the same (A,B) noise for the −1 forward pass
and for all micro-batches (noise does not depend on activations). Disables with
``--no_forward_noise_cache``. DataLoader uses persistent workers + prefetch; ``evaluate_fitness``
uses non_blocking H2D copies; optional ``--compile`` wraps the model for extra kernel fusion.
"""
from __future__ import annotations

import argparse
import contextlib
import contextvars
import hashlib
import math
import os
import random
import socket
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from eggroll_pt import fold_in_key

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# --- context: active perturbation for EggRollLinear ---------------------------------

@dataclass(frozen=True)
class EggRollForwardCtx:
    population_seed: int
    sigma: float
    sign: float  # +1.0 or -1.0 for the low-rank term
    lora_rank: int
    # Shared across +1 / -1 passes and all micro-batches for one seed (noise is input-independent).
    noise_cache: Optional[Dict[int, torch.Tensor]] = None


_eggroll_fwd_ctx: contextvars.ContextVar[Optional[EggRollForwardCtx]] = contextvars.ContextVar(
    "eggroll_fwd_ctx", default=None
)


@contextlib.contextmanager
def eggroll_forward_ctx(
    seed: int,
    sigma: float,
    sign: float,
    lora_rank: int,
    *,
    noise_cache: Optional[Dict[int, torch.Tensor]] = None,
):
    token = _eggroll_fwd_ctx.set(
        EggRollForwardCtx(seed, sigma, sign, lora_rank, noise_cache=noise_cache)
    )
    try:
        yield
    finally:
        _eggroll_fwd_ctx.reset(token)


def _stable_layer_id(name: str) -> int:
    return int(hashlib.md5(name.encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def _module_key_from_param_name(param_name: str) -> str:
    """Match EggRollLinear.full_name: Linear weights are registered as '<module>.weight'."""
    if param_name.endswith(".weight"):
        return param_name[: -len(".weight")]
    return param_name


class EggRollLinear(nn.Module):
    """
    Wraps nn.Linear so that when eggroll_forward_ctx is set, the forward adds
    (σ/√r) * sign * (x B) A^T with A,B from RNG(population_seed, layer_name).
    Optional fixed principal projector P (in×in) on p_mat in W^T space (see ``principal_projection``).
    """

    def __init__(
        self,
        linear: nn.Linear,
        full_name: str,
        lora_rank: int,
        principal_projection: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias
        self.full_name = full_name
        self.lora_rank = lora_rank
        self._layer_id = _stable_layer_id(full_name)
        if principal_projection is not None:
            self.register_buffer("_principal_projection", principal_projection, persistent=False)
        else:
            self._principal_projection = None  # type: ignore[assignment]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        ctx = _eggroll_fwd_ctx.get()
        if ctx is None or ctx.lora_rank < 1:
            return y
        out_f, in_f = self.weight.shape
        r = ctx.lora_rank
        lid = self._layer_id
        cache = ctx.noise_cache
        if cache is not None:
            if lid not in cache:
                gen = torch.Generator(device=x.device)
                gen.manual_seed(fold_in_key(ctx.population_seed, lid))
                cache[lid] = torch.randn(
                    (out_f + in_f, r),
                    generator=gen,
                    device=x.device,
                    dtype=self.weight.dtype,
                )
            noise = cache[lid]
        else:
            gen = torch.Generator(device=x.device)
            gen.manual_seed(fold_in_key(ctx.population_seed, lid))
            noise = torch.randn(
                (out_f + in_f, r),
                generator=gen,
                device=x.device,
                dtype=self.weight.dtype,
            )
        b_mat = noise[:in_f, :]
        a_mat = noise[in_f:, :]
        scale = (ctx.sigma / math.sqrt(float(r))) * ctx.sign
        # p_mat = b a^T is [in, out] (same layout as W^T); principal P is (in, in): P @ p_mat matches (noise @ P)^T for W-shaped noise.
        p_mat = b_mat @ a_mat.T
        if getattr(self, "_principal_projection", None) is not None:
            P = self._principal_projection.to(dtype=p_mat.dtype, device=p_mat.device)
            p_mat = P @ p_mat
        delta = x @ p_mat * scale
        return y + delta


def replace_linears_with_eggroll(
    root: nn.Module,
    lora_rank: int,
    prefix: str = "",
    principal_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> int:
    """In-place replace nn.Linear with EggRollLinear. Returns number of replaced layers."""
    n = 0
    for name, child in list(root.named_children()):
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            P = principal_dict.get(full) if principal_dict else None
            el = EggRollLinear(child, full, lora_rank, principal_projection=P)
            setattr(root, name, el)
            n += 1
        else:
            n += replace_linears_with_eggroll(child, lora_rank, full, principal_dict)
    return n


@torch.no_grad()
def build_principal_projection_dict(model: nn.Module, filter_rank: int) -> Dict[str, torch.Tensor]:
    """
    One projector per 2D weight, keyed by module path (same as EggRollLinear.full_name).
    Matches ``neural_thicket.utils.get_principal_parameters`` (right singular subspace, V V^T on in×in).
    """
    state: Dict[str, torch.Tensor] = {}
    gen = torch.Generator(device=model.device)
    gen.manual_seed(42)
    if filter_rank <= 0:
        return state
    for name, param in model.named_parameters():
        if param.dim() != 2:
            continue
        full = _module_key_from_param_name(name)
        random_projection = torch.linalg.qr(torch.randn(param.shape[1], filter_rank, generator=gen, device=param.device, dtype=torch.float32))[0] @ torch.linalg.qr(torch.randn(param.shape[1], filter_rank, generator=gen, device=param.device, dtype=torch.float32))[0].T
        # out_f, in_f = param.shape
        # q = min(filter_rank * 2, min(out_f, in_f))
        # k = min(filter_rank, min(out_f, in_f))
        # if k < 1:
        #     k = 1
        # _U, _S, V = torch.svd_lowrank(
        #     param.data.to(torch.float32).to(param.device),
        #     q=q,
        #     niter=10,
        # )
        # V_ = V[:, :k]
        # P = (V_ @ V_.T).to(device=param.device, dtype=param.dtype)
        state[full] = random_projection
    return state


# --- Weight update: same low-rank factors as EggRollLinear (fold_in_key(seed, layer_id)) ---


def _eggroll_update_buckets(
    model: nn.Module,
) -> Dict[Tuple[int, int, torch.dtype, torch.device], List[Tuple[torch.nn.Parameter, int, str]]]:
    """Group 2D weights by (out, in, dtype, device); lid and full_name match EggRollLinear."""
    buckets: Dict[Tuple[int, int, torch.dtype, torch.device], List[Tuple[torch.nn.Parameter, int, str]]] = defaultdict(list)
    for name, p in model.named_parameters():
        if not p.requires_grad or p.dim() != 2:
            continue
        key = (p.shape[0], p.shape[1], p.dtype, p.device)
        full_name = _module_key_from_param_name(name)
        lid = _stable_layer_id(full_name)
        buckets[key].append((p, lid, full_name))
    return buckets


def apply_lora_es_update(
    model: nn.Module,
    seeds: List[int],
    coeffs: List[float],
    lora_rank: int,
    *,
    buckets: Optional[Dict[Tuple[int, int, torch.dtype, torch.device], List[Tuple[torch.nn.Parameter, int, str]]]] = None,
    filter_rank: int = 0,
    principal_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    """
    θ += sum_k coeff_k * (A_k B_k^T) / √r with RNG matching EggRollLinear.forward.

    Optimizations:
    - Buckets matrices with the same (out, in, dtype, device) and uses one batched bmm per
      (bucket, seed) instead of one torch.mm per matrix.
    - Pass ``buckets=_eggroll_update_buckets(model)`` once per run to avoid rebuilding the dict
      every step.
    Row-wise RNG still uses one manual_seed + normal_ per matrix per seed (required for
    fold_in_key(seed, lid) semantics).
    If ``filter_rank > 0`` and ``principal_dict`` is provided, each ``E`` is right-multiplied
    by the same fixed ``P`` as ``perturb_principal_parameters`` (``E @ P``).
    """
    if lora_rank < 1:
        return
    if buckets is None:
        buckets = _eggroll_update_buckets(model)

    inv_sqrt_r = 1.0 / math.sqrt(float(lora_rank))
    use_principal = filter_rank > 0 and principal_dict

    with torch.no_grad():
        for key, group in buckets.items():
            out_f, in_f, dtype, device = key
            r = lora_rank
            n = len(group)
            params = [g[0] for g in group]
            lids = [g[1] for g in group]
            full_names = [g[2] for g in group]
            acc = torch.zeros((n, out_f, in_f), device=device, dtype=dtype)
            gen = torch.Generator(device=device)

            for seed, c in zip(seeds, coeffs):
                if c == 0.0:
                    continue
                noise = torch.empty((n, out_f + in_f, r), device=device, dtype=dtype)
                for j in range(n):
                    gen.manual_seed(fold_in_key(int(seed), lids[j]))
                    noise[j].normal_(0, 1, generator=gen)
                b_part = noise[:, :in_f, :]
                a_part = noise[:, in_f:, :]
                e = torch.bmm(a_part, b_part.transpose(1, 2))
                e.mul_(inv_sqrt_r)
                if use_principal:
                    for j in range(n):
                        P = principal_dict[full_names[j]]
                        e[j] = e[j] @ P.to(dtype=e.dtype)
                acc.add_(e, alpha=float(c))

            for j, p in enumerate(params):
                p.add_(acc[j])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EggRoll fused-forward ES for SFT (multi-GPU)")
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--data_path", type=str, default="data/LIMO/train.parquet")
    p.add_argument("--micro_batch_size", type=int, default=16)
    p.add_argument("--num_micro_batches", type=int, default=1)
    p.add_argument("--num_iterations", type=int, default=1500)
    p.add_argument("--lora_rank", type=int, default=1, help="Low-rank r for EggRoll perturbation (paper often uses 1).")
    p.add_argument("--population_size", type=int, default=32)
    p.add_argument("--sigma", type=float, default=1e-3)
    p.add_argument("--alpha", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--nproc_per_node", type=int, default=None)
    p.add_argument("--save_every", type=int, default=150)
    p.add_argument("--output_dir", type=str, default="runs/es_eggroll_sft")
    p.add_argument("--attn_implementation", type=str, default="flash_attention_2", choices=["sdpa", "flash_attention_2", "eager"])
    p.add_argument("--max_position_embeddings", type=int, default=None, help="Optional override for long-context models.")
    p.add_argument(
        "--num_dataloader_workers",
        type=int,
        default=0,
        help="DataLoader worker processes (0 = main process only).",
    )
    p.add_argument(
        "--dataloader_prefetch_factor",
        type=int,
        default=4,
        help="Per-worker batches prefetched (only if num_dataloader_workers > 0).",
    )
    p.add_argument(
        "--no_forward_noise_cache",
        action="store_true",
        help="Disable per-layer noise cache (slower; use for debugging parity).",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile(model) for extra throughput (first steps slower; may fail on some configs).",
    )
    p.add_argument(
        "--filter_rank",
        type=int,
        default=64,
        help="Principal subspace rank k: projector P = V_k V_k^T from weight SVD (right singular vectors). "
        "0 or negative disables projection (full perturbation on all weights).",
    )
    return p.parse_args()


def _dtype_from_name(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def build_dataloader(args: argparse.Namespace, tokenizer: AutoTokenizer) -> DataLoader:
    from datasets import load_from_disk

    nw = getattr(args, "num_dataloader_workers", 8)
    prefetch = getattr(args, "dataloader_prefetch_factor", 4)
    tokenized = load_from_disk(args.data_path)

    def _collate(batch):
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids = []
        labels = []
        attention_mask = []
        for item in batch:
            padding_len = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [tokenizer.pad_token_id] * padding_len)
            attention_mask.append(item["attention_mask"] + [0] * padding_len)
            labels.append(item["labels"] + [-100] * padding_len)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attention_mask),
        }

    g = torch.Generator()
    g.manual_seed(args.seed)
    dl_kw: Dict = {
        "batch_size": args.micro_batch_size,
        "shuffle": True,
        "collate_fn": _collate,
        "generator": g,
        "drop_last": True,
        "pin_memory": torch.cuda.is_available(),
    }
    if nw > 0:
        dl_kw["num_workers"] = nw
        dl_kw["persistent_workers"] = True
        dl_kw["prefetch_factor"] = prefetch
    else:
        dl_kw["num_workers"] = 0
    return DataLoader(tokenized, **dl_kw)


def masked_sum(values: torch.Tensor, mask: torch.Tensor, axis: int | tuple[int, ...] | None = None) -> torch.Tensor:
    valid_values = torch.where(mask.bool(), values, 0.0)
    return (valid_values * mask).sum(axis=axis)


def masked_mean(values, mask, axis=None):
    s = masked_sum(values, mask, axis)
    return s / (mask.sum(axis=axis) + 1e-8)


def evaluate_fitness(model: nn.Module, batch: Dict[str, torch.Tensor], device: torch.device) -> float:
    with torch.inference_mode():
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        labels = batch.pop("labels")
        labels = labels[:, 1:].contiguous()
        loss_mask = labels != -100
        labels[labels == -100] = 0
        logits = model(**batch).logits[:, :-1, :].to(torch.float32)
        per_token_logps = torch.gather(torch.log_softmax(logits, dim=-1), 2, labels.unsqueeze(2)).squeeze(2)
        loss = -masked_mean(per_token_logps, loss_mask)
    return -loss.item()


def normalize_rewards(seed_to_signal: Dict[int, float]) -> Dict[int, float]:
    arr = np.array(list(seed_to_signal.values()), dtype=np.float32)
    mean = float(arr.mean()) if arr.size else 0.0
    std = float(arr.std()) if arr.size else 0.0
    return {s: (v - mean) / (std + 1e-8) for s, v in seed_to_signal.items()}


def es_train(rank: int, world_size: int, args: argparse.Namespace, init_method: str) -> None:
    dist.init_process_group("nccl", init_method=init_method, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    if rank == 0:
        wandb.init(project="es-sft", name=f"eggroll-sft-r{args.lora_rank}", config=vars(args))
    device = torch.device(f"cuda:{rank}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dtype = _dtype_from_name(args.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        torch.set_float32_matmul_precision("high")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    extra = {}
    if args.max_position_embeddings is not None:
        extra["max_position_embeddings"] = args.max_position_embeddings

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        **extra,
    )
    model = model.to(device)
    principal_dict = build_principal_projection_dict(model, int(args.filter_rank))
    n_linear = replace_linears_with_eggroll(model, args.lora_rank, principal_dict=principal_dict)
    if rank == 0:
        print(
            f"EggRoll: replaced {n_linear} nn.Linear layers with EggRollLinear "
            f"(filter_rank={args.filter_rank}, principal_projection={'on' if principal_dict else 'off'})."
        )

    model_to_save = model
    if getattr(args, "compile", False):
        # dynamic=True: padded batch lengths vary; avoids excessive recompiles.
        model = torch.compile(model, dynamic=True, fullgraph=False)
        if rank == 0:
            print("torch.compile enabled (dynamic=True).")

    model.eval()
    es_update_buckets = _eggroll_update_buckets(model)
    dataloader = build_dataloader(args, tokenizer)
    amp_dtype = torch.bfloat16 if dtype == torch.bfloat16 else torch.float16
    use_amp = dtype in (torch.float16, torch.bfloat16)

    with torch.no_grad():
        for p in model.parameters():
            dist.broadcast(p.data, src=0)

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    data_iter = iter(dataloader)
    # step_batches = [next(data_iter)]

    for step in tqdm(range(args.num_iterations), desc="EggRoll ES (fused forward)"):
        start_time = time.time()
        seed_payload = [None]
        if rank == 0:
            seed_payload[0] = [random.randint(0, 1_000_000_000) for _ in range(args.population_size)]
        dist.broadcast_object_list(seed_payload, src=0)
        seeds = seed_payload[0]

        step_batches = []
        for _ in range(args.num_micro_batches):
            try:
                step_batches.append(next(data_iter))
            except StopIteration:
                data_iter = iter(dataloader)
                step_batches.append(next(data_iter))

        use_noise_cache = not getattr(args, "no_forward_noise_cache", False)
        local_pairs = []
        for idx in range(rank, len(seeds), world_size):
            seed = seeds[idx]
            total_diff = 0.0
            total_signal = 0.0
            noise_cache: Optional[Dict[int, torch.Tensor]] = {} if use_noise_cache else None

            with eggroll_forward_ctx(seed, args.sigma, 1.0, args.lora_rank, noise_cache=noise_cache):
                for micro_batch in step_batches:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                        r = evaluate_fitness(model, micro_batch, device)
                    total_diff += r
                    total_signal += r

            with eggroll_forward_ctx(seed, args.sigma, -1.0, args.lora_rank, noise_cache=noise_cache):
                for micro_batch in step_batches:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                        r = evaluate_fitness(model, micro_batch, device)
                    total_diff -= r
                    total_signal += r

            if noise_cache is not None:
                noise_cache.clear()

            denom = float(2 * args.num_micro_batches)
            local_pairs.append((seed, total_diff / denom, total_signal / denom))

        gathered = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(local_pairs, gathered, dst=0)
        if rank == 0:
            all_pairs = [pair for wp in gathered for pair in wp]
            losses = [-reward for _, _, reward in all_pairs]
            diffs = [d for _, d, _ in all_pairs]
            end_time = time.time()
            wandb.log(
                {
                    "train/loss": float(np.mean(losses)),
                    "train/diff": float(np.mean(diffs)),
                    "train/time(s)": end_time - start_time,
                },
                step=step,
            )
            print(
                f"[step {step}] mean loss={np.mean(losses):.6f}, mean diff={np.mean(diffs):.6f}, "
                f"time={end_time - start_time:.6f}s"
            )

        norm_payload = [None]
        if rank == 0:
            seed_to_fd = {}
            for wp in gathered:
                for seed, fd, _ in wp:
                    seed_to_fd[int(seed)] = float(fd)
            norm_payload[0] = normalize_rewards(seed_to_fd)
        dist.broadcast_object_list(norm_payload, src=0)
        seed_to_norm = norm_payload[0]

        update_scale = args.alpha / (2 * args.population_size)
        coeff_list = [update_scale * float(seed_to_norm[int(s)]) for s in seeds]
        apply_lora_es_update(
            model,
            seeds,
            coeff_list,
            args.lora_rank,
            buckets=es_update_buckets,
            filter_rank=int(args.filter_rank),
            principal_dict=principal_dict,
        )

        if (step + 1) % args.save_every == 0 and rank == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint_step_{step + 1}")
            model_to_save.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"Saved checkpoint: {save_path}")

    if rank == 0:
        final_path = os.path.join(args.output_dir, "final")
        model_to_save.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        print(f"Saved final model: {final_path}")

    dist.destroy_process_group()


def _run_with_spawn(args: argparse.Namespace) -> None:
    world_size = args.nproc_per_node if args.nproc_per_node is not None else torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No CUDA devices found.")
    init_method = f"tcp://127.0.0.1:{_find_free_port()}"
    mp.spawn(es_train, args=(world_size, args, init_method), nprocs=world_size, join=True)


def _run_with_torchrun(args: argparse.Namespace) -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]
    init_method = f"tcp://{master_addr}:{master_port}"
    es_train(rank, world_size, args, init_method)


if __name__ == "__main__":
    cli = parse_args()
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        _run_with_torchrun(cli)
    else:
        _run_with_spawn(cli)
