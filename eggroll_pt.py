"""
PyTorch port of the JAX EggRoll noiser (group ES, LoRA / full / noop maps).

Mirrors the structure of the JAX EggRoll class: same hyperparameters, RNG semantics
via deterministic seeds (equivalent to jax.random.fold_in chains), convert_fitnesses,
and grad estimates passed to torch.optim.SGD (optax.sgd equivalent).

Thread-parallel parts of _simple_*_update are vectorized (JAX vmap equivalent).
`torch.func.vmap` is used for the pure LoRA row split (A/B from stacked noise).
Per-thread RNG in the JAX reference uses independent subkeys; the vectorized path
draws a single `randn(N, ...)` block (IID standard normals), which matches the
marginal distribution used in ES. For bitwise parity with per-row `fold_in` seeds,
use the scalar `get_lora_update_params` / `get_nonlora_update_params` in a loop.
"""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

try:
    from torch.func import vmap as _vmap
except ImportError:
    from functorch import vmap as _vmap  # type: ignore[no-redef]

# functorch vmap (used for pure LoRA row split); public for callers who extend EggRoll
vmap = _vmap

# es_map classification (same indices as JAX EggRoll)
FULL = 0
LORA = 1
NOOP = 2
NOOP_ALT = 3


def _stable_string_id(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def _params_for_optimizer(params: Any) -> Any:
    """SGD/Adam expect an iterable of tensors; accept dict pytrees like JAX."""
    if isinstance(params, dict):
        return list(params.values())
    return params


def fold_in_key(base_key: int, *parts: int) -> int:
    """Deterministic analogue of repeated jax.random.fold_in."""
    h = int(base_key) & 0xFFFFFFFF
    for p in parts:
        h = (h ^ (int(p) * 1315423911)) & 0xFFFFFFFF
    return h & 0x7FFFFFFF


def get_lora_update_params(
    frozen_noiser_params: Dict[str, Any],
    base_sigma: float,
    iterinfo: Tuple[int, int],
    param: torch.Tensor,
    key: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    epoch, thread_id = iterinfo
    nr = int(frozen_noiser_params["noise_reuse"])
    true_epoch = 0 if nr == 0 else epoch // nr
    true_thread_idx = thread_id // 2
    sigma = base_sigma if (thread_id % 2 == 0) else -base_sigma

    a, b = param.shape
    r = int(frozen_noiser_params["rank"])
    device, dtype = param.device, param.dtype

    gen = torch.Generator(device=device)
    gen.manual_seed(fold_in_key(key, true_epoch, true_thread_idx))
    lora_params = torch.randn((a + b, r), generator=gen, device=device, dtype=dtype)
    B_mat = lora_params[:b]
    A_mat = lora_params[b:]
    return A_mat * sigma, B_mat


def get_nonlora_update_params(
    frozen_noiser_params: Dict[str, Any],
    base_sigma: float,
    iterinfo: Tuple[int, int],
    param: torch.Tensor,
    key: int,
) -> torch.Tensor:
    epoch, thread_id = iterinfo
    nr = int(frozen_noiser_params["noise_reuse"])
    true_epoch = 0 if nr == 0 else epoch // nr
    true_thread_idx = thread_id // 2
    sigma = base_sigma if (thread_id % 2 == 0) else -base_sigma

    device, dtype = param.device, param.dtype
    gen = torch.Generator(device=device)
    gen.manual_seed(fold_in_key(key, true_epoch, true_thread_idx))
    updates = torch.randn(param.shape, generator=gen, device=device, dtype=dtype)
    return updates * sigma


def _simple_full_update(
    base_sigma: float,
    param: torch.Tensor,
    key: int,
    scores: torch.Tensor,
    iterinfos: Tuple[torch.Tensor, torch.Tensor],
    frozen_noiser_params: Dict[str, Any],
) -> torch.Tensor:
    """Vectorized over threads (JAX: vmap(get_nonlora_update_params) + mean(scores * updates))."""
    if frozen_noiser_params["freeze_nonlora"]:
        return torch.zeros_like(param)
    epoch_vec, thread_ids = iterinfos
    n = scores.shape[0]
    device, dtype = param.device, param.dtype

    sigma_row = torch.where(
        (thread_ids % 2) == 0,
        torch.tensor(base_sigma, device=device, dtype=dtype),
        torch.tensor(-base_sigma, device=device, dtype=dtype),
    )
    sigma_row = sigma_row.reshape(n, *([1] * param.ndim))

    gen = torch.Generator(device=device)
    gen.manual_seed(fold_in_key(key, 0, 0))
    eps = torch.randn((n,) + param.shape, generator=gen, device=device, dtype=dtype)
    upd = eps * sigma_row
    w = scores.reshape(n, *([1] * param.ndim)).to(dtype)
    accum = (w * upd).sum(dim=0) / n
    return accum.to(param.dtype)


def _simple_lora_update(
    base_sigma: float,
    param: torch.Tensor,
    key: int,
    scores: torch.Tensor,
    iterinfos: Tuple[torch.Tensor, torch.Tensor],
    frozen_noiser_params: Dict[str, Any],
) -> torch.Tensor:
    """Vectorized over threads + vmap for pure A/B split (JAX: vmap(get_lora_update_params) + einsum)."""
    rank = int(frozen_noiser_params["rank"])
    sigma_lora = base_sigma / math.sqrt(rank)
    _, thread_ids = iterinfos
    n = scores.shape[0]
    a, b = param.shape
    r = rank
    device, dtype = param.device, param.dtype

    sigma_row = torch.where(
        (thread_ids % 2) == 0,
        torch.tensor(sigma_lora, device=device, dtype=dtype),
        torch.tensor(-sigma_lora, device=device, dtype=dtype),
    )

    gen = torch.Generator(device=device)
    gen.manual_seed(fold_in_key(key, 0, 0))
    lora_params = torch.randn((n, a + b, r), generator=gen, device=device, dtype=dtype)

    def _split_lora(lp: torch.Tensor, sig: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = b
        B_mat = lp[:bsz]
        A_mat = lp[bsz:] * sig
        return A_mat, B_mat

    Am, Bm = vmap(_split_lora)(lora_params, sigma_row)

    s = scores.view(n, 1, 1).to(dtype)
    A_weighted = s * Am
    out = torch.einsum("nir,njr->ij", A_weighted, Bm) / n
    return out.to(param.dtype)


def _noop_update(
    base_sigma: float,
    param: torch.Tensor,
    key: int,
    scores: torch.Tensor,
    iterinfos: Tuple[torch.Tensor, torch.Tensor],
    frozen_noiser_params: Dict[str, Any],
) -> torch.Tensor:
    return torch.zeros_like(param)


_UPDATE_FNS = (_simple_full_update, _simple_lora_update, _noop_update, _noop_update)


def convert_fitnesses(
    frozen_noiser_params: Dict[str, Any],
    noiser_params: Dict[str, Any],
    raw_scores: torch.Tensor,
    num_episodes_list: Optional[Any] = None,
) -> torch.Tensor:
    del noiser_params, num_episodes_list  # API parity with JAX
    group_size = int(frozen_noiser_params["group_size"])
    eps = 1e-5
    if group_size == 0:
        mean = raw_scores.mean()
        var = raw_scores.var(unbiased=False)
        true_scores = (raw_scores - mean) / torch.sqrt(var + eps)
    else:
        group_scores = raw_scores.reshape(-1, group_size)
        mean_g = group_scores.mean(dim=-1, keepdim=True)
        # JAX uses jnp.var(raw_scores) globally (not per-group)
        var = raw_scores.var(unbiased=False)
        true_scores = (group_scores - mean_g) / torch.sqrt(var + eps)
        true_scores = true_scores.reshape(-1)
    return true_scores


class EggRoll:
    """PyTorch analogue of JAX EggRoll Noiser."""

    @classmethod
    def init_noiser(
        cls,
        params: Any,
        sigma: float,
        lr: float,
        *args: Any,
        solver: Optional[Callable[..., torch.optim.Optimizer]] = None,
        solver_kwargs: Optional[Dict[str, Any]] = None,
        group_size: int = 0,
        freeze_nonlora: bool = False,
        noise_reuse: int = 0,
        rank: int = 1,
        use_batched_update: bool = False,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        del args, kwargs
        if solver is None:
            solver = torch.optim.SGD
        if solver_kwargs is None:
            solver_kwargs = {}
        opt = solver(_params_for_optimizer(params), lr=lr, **solver_kwargs)
        frozen = {
            "group_size": group_size,
            "freeze_nonlora": freeze_nonlora,
            "noise_reuse": noise_reuse,
            "solver": opt,
            "rank": rank,
            "use_batched_update": use_batched_update,
        }
        noiser_state = {"sigma": sigma}
        return frozen, noiser_state

    @classmethod
    def do_mm(
        cls,
        frozen_noiser_params: Dict[str, Any],
        noiser_params: Dict[str, Any],
        param: torch.Tensor,
        base_key: int,
        iterinfo: Optional[Tuple[int, int]],
        x: torch.Tensor,
    ) -> torch.Tensor:
        base_ans = x @ param.T
        if iterinfo is None:
            return base_ans
        rank = int(frozen_noiser_params["rank"])
        sig = float(noiser_params["sigma"]) / math.sqrt(rank)
        A, B = get_lora_update_params(frozen_noiser_params, sig, iterinfo, param, base_key)
        return base_ans + x @ B @ A.T

    @classmethod
    def do_Tmm(
        cls,
        frozen_noiser_params: Dict[str, Any],
        noiser_params: Dict[str, Any],
        param: torch.Tensor,
        base_key: int,
        iterinfo: Optional[Tuple[int, int]],
        x: torch.Tensor,
    ) -> torch.Tensor:
        base_ans = x @ param
        if iterinfo is None:
            return base_ans
        rank = int(frozen_noiser_params["rank"])
        sig = float(noiser_params["sigma"]) / math.sqrt(rank)
        A, B = get_lora_update_params(frozen_noiser_params, sig, iterinfo, param, base_key)
        return base_ans + x @ A @ B.T

    @classmethod
    def do_emb(
        cls,
        frozen_noiser_params: Dict[str, Any],
        noiser_params: Dict[str, Any],
        param: torch.Tensor,
        base_key: int,
        iterinfo: Optional[Tuple[int, int]],
        x: torch.Tensor,
    ) -> torch.Tensor:
        del frozen_noiser_params, noiser_params, param, base_key, iterinfo, x
        raise NotImplementedError("Embedding is not implemented")

    @classmethod
    def get_noisy_standard(
        cls,
        frozen_noiser_params: Dict[str, Any],
        noiser_params: Dict[str, Any],
        param: torch.Tensor,
        base_key: int,
        iterinfo: Optional[Tuple[int, int]],
    ) -> torch.Tensor:
        if iterinfo is None or frozen_noiser_params["freeze_nonlora"]:
            return param
        return param + get_nonlora_update_params(
            frozen_noiser_params, float(noiser_params["sigma"]), iterinfo, param, base_key
        )

    @classmethod
    def _do_update(
        cls,
        param: torch.Tensor,
        base_key: int,
        fitnesses: torch.Tensor,
        iterinfos: Tuple[torch.Tensor, torch.Tensor],
        map_classification: int,
        sigma: float,
        frozen_noiser_params: Dict[str, Any],
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        update_fn = _UPDATE_FNS[map_classification]
        new_grad = update_fn(sigma, param, base_key, fitnesses, iterinfos, frozen_noiser_params)
        n = fitnesses.numel()
        return -(new_grad * math.sqrt(n)).to(param.dtype)

    @classmethod
    def do_updates(
        cls,
        frozen_noiser_params: Dict[str, Any],
        noiser_params: Dict[str, Any],
        params: Any,
        base_keys: Any,
        fitnesses: torch.Tensor,
        iterinfos: Tuple[torch.Tensor, torch.Tensor],
        es_map: Any,
    ) -> Tuple[Dict[str, Any], Any]:
        if frozen_noiser_params["use_batched_update"]:
            return cls._do_updates_batched(
                frozen_noiser_params, noiser_params, params, base_keys, fitnesses, iterinfos, es_map
            )
        return cls._do_updates_original(
            frozen_noiser_params, noiser_params, params, base_keys, fitnesses, iterinfos, es_map
        )

    @classmethod
    def _do_updates_original(
        cls,
        frozen_noiser_params: Dict[str, Any],
        noiser_params: Dict[str, Any],
        params: Any,
        base_keys: Any,
        fitnesses: torch.Tensor,
        iterinfos: Tuple[torch.Tensor, torch.Tensor],
        es_map: Any,
    ) -> Tuple[Dict[str, Any], Any]:
        sigma = float(noiser_params["sigma"])
        opt: torch.optim.Optimizer = frozen_noiser_params["solver"]

        def one_leaf(p: torch.Tensor, k: Union[int, torch.Tensor], m: int) -> torch.Tensor:
            # Scalar key: param is a single tensor. Stacked keys: param has leading dim K matching key[j] (JAX lax.scan).
            if isinstance(k, torch.Tensor) and k.ndim > 0:
                outs = []
                for j in range(k.shape[0]):
                    outs.append(
                        cls._do_update(p[j], int(k[j].item()), fitnesses, iterinfos, m, sigma, frozen_noiser_params)
                    )
                return torch.stack(outs, dim=0)
            return cls._do_update(p, int(k), fitnesses, iterinfos, m, sigma, frozen_noiser_params)

        new_grad = map_params_tree(one_leaf, params, base_keys, es_map)

        opt.zero_grad()
        for tp, g in iter_params_and_grads(params, new_grad):
            if tp.grad is None:
                tp.grad = g.detach().clone()
            else:
                tp.grad.copy_(g)
        opt.step()
        return noiser_params, params

    @classmethod
    def _do_updates_batched(
        cls,
        frozen_noiser_params: Dict[str, Any],
        noiser_params: Dict[str, Any],
        params: Any,
        base_keys: Any,
        fitnesses: torch.Tensor,
        iterinfos: Tuple[torch.Tensor, torch.Tensor],
        es_map: Any,
    ) -> Tuple[Dict[str, Any], Any]:
        sigma = float(noiser_params["sigma"])
        opt: torch.optim.Optimizer = frozen_noiser_params["solver"]

        flat_params, treedef = tree_flatten(params)
        flat_keys, _ = tree_flatten(base_keys)
        flat_es, _ = tree_flatten(es_map)

        buckets: Dict[Tuple[Any, int], List[int]] = defaultdict(list)
        for i, (param, map_class) in enumerate(zip(flat_params, flat_es)):
            key = (param.shape, int(map_class))
            buckets[key].append(i)

        new_flat_grads: List[Optional[torch.Tensor]] = [None] * len(flat_params)

        # Python loop over same-shape layers: vmap would require RNG without int(key)/.item()
        # (functorch limitation); each _do_update is already thread-vectorized inside.
        for (_, map_class), indices in buckets.items():
            stacked_params = torch.stack([flat_params[i] for i in indices], dim=0)
            key_list = [flat_keys[i] for i in indices]
            if isinstance(key_list[0], torch.Tensor) and key_list[0].ndim > 0:
                stacked_keys = torch.stack(key_list, dim=0)
            else:
                stacked_keys = torch.tensor([int(x) for x in key_list], device=stacked_params.device)

            grads_batch = torch.stack(
                [
                    cls._do_update(
                        stacked_params[j],
                        int(stacked_keys[j].item()),
                        fitnesses,
                        iterinfos,
                        map_class,
                        sigma,
                        frozen_noiser_params,
                    )
                    for j in range(stacked_params.shape[0])
                ],
                dim=0,
            )
            for k, idx in enumerate(indices):
                new_flat_grads[idx] = grads_batch[k]

        new_grad = tree_unflatten(treedef, new_flat_grads)

        opt.zero_grad()
        for tp, g in iter_params_and_grads(params, new_grad):
            if tp.grad is None:
                tp.grad = g.detach().clone()
            else:
                tp.grad.copy_(g)
        opt.step()
        return noiser_params, params


# --- minimal pytree helpers (JAX tree_flatten / tree_unflatten analogue) ---


@dataclass
class _TreeDef:
    type_name: str
    children: List[Any]


def tree_flatten(tree: Any) -> Tuple[List[Any], _TreeDef]:
    if isinstance(tree, dict):
        keys = sorted(tree.keys())
        flat = [tree[k] for k in keys]
        return flat, _TreeDef("dict", keys)
    if isinstance(tree, (list, tuple)):
        flat = list(tree)
        return flat, _TreeDef("tuple" if isinstance(tree, tuple) else "list", len(tree))
    return [tree], _TreeDef("leaf", None)


def tree_unflatten(treedef: _TreeDef, flat: List[Any]) -> Any:
    if treedef.type_name == "dict":
        return {k: v for k, v in zip(treedef.children, flat)}
    if treedef.type_name == "list":
        return list(flat)
    if treedef.type_name == "tuple":
        return tuple(flat)
    return flat[0]


def map_params_tree(
    fn: Callable[..., torch.Tensor],
    params: Any,
    base_keys: Any,
    es_map: Any,
) -> Any:
    if isinstance(params, dict):
        return {
            k: fn(params[k], base_keys[k], es_map[k])
            for k in params
        }
    if isinstance(params, (list, tuple)):
        out = [fn(params[i], base_keys[i], es_map[i]) for i in range(len(params))]
        return tuple(out) if isinstance(params, tuple) else out
    return fn(params, base_keys, es_map)


def iter_params_and_grads(params: Any, grads: Any):
    if isinstance(params, dict):
        for k in params:
            yield from iter_params_and_grads(params[k], grads[k])
    elif isinstance(params, (list, tuple)):
        for i in range(len(params)):
            yield from iter_params_and_grads(params[i], grads[i])
    else:
        yield params, grads


def params_from_module(module: nn.Module) -> List[nn.Parameter]:
    return [p for p in module.parameters() if p.requires_grad]


def example_es_map_for_module(module: nn.Module, freeze_nonlora: bool = False) -> Dict[str, int]:
    """Name -> map class; 2D weights LORA, 1D FULL or NOOP."""
    m: Dict[str, int] = {}
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() == 2:
            m[name] = LORA
        elif p.dim() == 1:
            m[name] = NOOP if freeze_nonlora else FULL
        else:
            m[name] = NOOP
    return m


def build_param_and_key_dicts(
    module: nn.Module,
    es_map_by_name: Dict[str, int],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int], Dict[str, int]]:
    """Flat dicts keyed by parameter name for use with EggRoll.do_updates."""
    params_d: Dict[str, torch.Tensor] = {}
    keys_d: Dict[str, int] = {}
    map_d: Dict[str, int] = {}
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        params_d[name] = p
        keys_d[name] = _stable_string_id(name)
        map_d[name] = int(es_map_by_name.get(name, NOOP))
    return params_d, keys_d, map_d
