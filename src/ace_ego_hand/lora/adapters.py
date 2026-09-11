"""Low-rank adaptation of frozen linear layers: loaded from a delta file, merged or live.

LoRA keeps a pretrained layer frozen and adds a low-rank residual `scaling * B(A(x))` beside it.
ACE-Ego-Hand fine-tuned the Wan trunk this way (rank 64, alpha 64, every projection of the
attention and feed-forward layers) and ships the adapters separately from the base weights.
`apply_lora` swaps the targeted `nn.Linear` layers of a model for `LoRALinear` in place, so the
model object keeps its identity and its `state_dict` names, and returns a `LoRAModel` handle
whose only jobs are loading the adapter tensors and toggling or merging them. The estimator
merges them into the bf16 base weights at load (`ModelConfig.merge_lora`, one matmul per
linear); kept live, the residual is computed in the base layer's dtype (bf16 in the trunk), as
the upstream inference path does under autocast. Inference only: no dropout, and the adapters
are always loaded from a file.
Ref: LoRA arXiv:2106.09685
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import final, override

import torch
from torch import nn

from ..types import Activations, WeightTensor

LORA_A = "lora_A"
LORA_B = "lora_B"


@dataclass(frozen=True, slots=True)
class LoRAConfig:
    """Rank, scale and targets of the adapters, as read from a variant's `config.json`.

    `target_modules` are matched against `named_modules` names either exactly or as a dotted
    suffix (`feed_forward.0` matches `blocks.3.feed_forward.0`). Defaults are the ACE-Ego-Hand
    checkpoints.
    """

    rank: int = 64
    alpha: float = 64.0
    target_modules: tuple[str, ...] = (
        "self_attention.query",
        "self_attention.key",
        "self_attention.value",
        "self_attention.output",
        "cross_attention.query",
        "cross_attention.key",
        "cross_attention.value",
        "cross_attention.output",
        "feed_forward.0",
        "feed_forward.2",
    )

    @property
    def scaling(self) -> float:
        """The residual's multiplier `alpha / rank`."""

        return self.alpha / self.rank


class LoRAAdapter[BaseModuleT: nn.Module](nn.Module, ABC):
    """A frozen base module plus a low-rank residual that can be switched off or merged."""

    def __init__(self, base: BaseModuleT, scaling: float) -> None:
        super().__init__()
        self.base = base
        self.scaling = scaling
        self.enabled = True
        self.base.requires_grad_(False)

    @abstractmethod
    @override
    def forward(self, x: Activations) -> Activations: ...

    @abstractmethod
    def delta_weight(self) -> WeightTensor:
        """The residual expressed as an additive update of the base weight."""

    def merge(self) -> BaseModuleT:
        """Fold the residual into the base weight and return the base module.

        The sum is formed in float32 and rounded once to the weight's dtype, so a bf16 base
        weight ends up as `round(W + delta)` rather than `round(W + round(delta))`.
        """

        with torch.no_grad():
            weight = self.base.get_parameter("weight")
            weight.copy_((weight.float() + self.delta_weight().float()).to(weight.dtype))
        return self.base


@final
class LoRALinear(LoRAAdapter[nn.Linear]):
    """`nn.Linear` with a rank-`r` residual `scaling * lora_B(lora_A(x))`.

    The attribute names `lora_A` / `lora_B` are the checkpoint's tensor names, so a delta file
    loads without renaming.
    """

    def __init__(self, base: nn.Linear, config: LoRAConfig) -> None:
        super().__init__(base, config.scaling)
        device, dtype = base.weight.device, base.weight.dtype
        self.lora_A = nn.Linear(
            base.in_features, config.rank, bias=False, device=device, dtype=dtype
        )
        self.lora_B = nn.Linear(
            config.rank, base.out_features, bias=False, device=device, dtype=dtype
        )
        # A random, B zero: the residual is a no-op until the adapter tensors are loaded
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    @override
    def forward(self, x: Activations) -> Activations:
        out = self.base(x)
        if not self.enabled:
            return out
        return out + self.lora_B(self.lora_A(x)) * self.scaling

    @override
    def delta_weight(self) -> WeightTensor:
        # The product in float32: the rank-64 contraction should not round at bf16 twice
        return (self.lora_B.weight.float() @ self.lora_A.weight.float()) * self.scaling


class LoRAModel[ModelT: nn.Module](nn.Module):
    """Handle on a model whose targeted layers were swapped for adapters by `apply_lora`.

    The wrapped model is used directly for inference (its layers were replaced in place); this
    handle only addresses the adapters: loading their tensors, toggling and merging them.
    Adapter tensor names are the wrapped model's own `state_dict` names.
    """

    def __init__(self, model: ModelT) -> None:
        super().__init__()
        self.model = model

    def adapters(self) -> Iterator[tuple[str, LoRAAdapter[nn.Module]]]:
        """`(name, adapter)` for every adapter, named like `model.named_modules()`."""

        for name, module in self.model.named_modules():
            if isinstance(module, LoRAAdapter):
                yield name, module

    def set_lora_enabled(self, enabled: bool) -> None:
        for _, adapter in self.adapters():
            adapter.enabled = enabled

    def lora_state_dict(self) -> dict[str, WeightTensor]:
        """The adapter tensors, keyed like the wrapped model's `state_dict`."""

        return {
            name: value
            for name, value in self.model.state_dict().items()
            if is_lora_parameter_name(name)
        }

    @torch.no_grad()
    def load_lora_state_dict(self, state: Mapping[str, WeightTensor]) -> None:
        """Copy adapter tensors in, cast to the adapters' dtype; every adapter exactly once."""

        expected = self.lora_state_dict()
        unexpected = sorted(state.keys() - expected.keys())
        missing = sorted(expected.keys() - state.keys())
        if unexpected or missing:
            raise KeyError(
                f"LoRA state mismatch: unexpected {unexpected[:3]}, missing {missing[:3]}"
            )
        for name, value in state.items():
            target = expected[name]
            if value.shape != target.shape:
                raise ValueError(f"{name}: shape {tuple(value.shape)} != {tuple(target.shape)}")
            target.copy_(value.to(target.dtype))

    def merge(self) -> ModelT:
        """Fold every adapter into its base layer and return the plain model."""

        for name, adapter in list(self.adapters()):
            _replace_submodule(self.model, name, adapter.merge())
        return self.model


def apply_lora[ModelT: nn.Module](model: ModelT, config: LoRAConfig) -> LoRAModel[ModelT]:
    """Swap every targeted `nn.Linear` of `model` for a `LoRALinear`, in place.

    Fails if nothing matches: a silent no-op would mean the fine-tuned weights are never applied.
    """

    targets = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and _is_target(name, config)
    ]
    if not targets:
        raise ValueError(f"no module matches the LoRA targets {config.target_modules!r}")
    for name, module in targets:
        _replace_submodule(model, name, LoRALinear(module, config))
    return LoRAModel(model)


def is_lora_parameter_name(name: str) -> bool:
    """Whether a `state_dict` key belongs to an adapter rather than a base layer."""

    return f".{LORA_A}." in name or f".{LORA_B}." in name


def _is_target(name: str, config: LoRAConfig) -> bool:
    return any(name == target or name.endswith(f".{target}") for target in config.target_modules)


def _replace_submodule(model: nn.Module, name: str, replacement: nn.Module) -> None:
    parent_name, _, attribute = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, attribute, replacement)
