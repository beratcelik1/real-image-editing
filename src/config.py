"""YAML config loading utility.

Each `configs/*.yaml` file is loaded into a typed dataclass so config
errors surface at load time, not deep inside training loops. The
config schemas mirror the YAML files; extending either requires
editing both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Config schemas
# ---------------------------------------------------------------------------


@dataclass
class Pez1Config:
    """PEZ-1 source-image inversion config (configs/pez_1.yaml).

    See RESEARCH_PROPOSAL.md §3.1 for the alternating R=2 algorithm.
    Output is a continuous Tensor[prompt_length, 768] of CLIP-input
    embeddings (no vocabulary projection).
    """

    # Loss formulation
    loss_type: str  # "sds" or "clip"
    cfg_scale: float
    timestep_sampling: str  # "uniform" or "importance"

    # DDIM/null-text num_steps used in SDS rounds. Must match
    # EditConfig.ddim.num_steps so editing-time null-text-per-timestep
    # has the same length the editing loop consumes (otherwise
    # run_p2p_edit silently falls back to default uncond — see Bug #3).
    ddim_num_steps: int

    # Prompt structure
    prompt_length: int

    # Optimization
    num_steps: int
    learning_rate: float
    weight_decay: float            # round-0 vanilla-PEZ weight decay
                                   # (Wen-et-al default; decays soft_prompt
                                   # toward origin from random init).
    delta_weight_decay: float      # round-1+ anchor strength via the
                                   # residual parameterization
                                   # P = c_anchor + Δ (proposal §3.1).
                                   # AdamW.weight_decay applied to Δ.
    batch_size: int

    # Models
    clip_model: str

    # Alternating-optimization rounds
    num_rounds: int

    # Caching
    cache_dir: str
    use_cache: bool

    # Misc
    seed: int
    device: str
    dtype: str


@dataclass
class Pez2Config:
    """PEZ-2 instruction-conditioned generation config (configs/pez_2.yaml).

    See RESEARCH_PROPOSAL.md §3.2 for the three-term loss. Output is
    a continuous Tensor[prompt_length, 768] in the same shape as PEZ-1.
    """

    source_loss_type: str  # "sds" or "clip"; should match Pez1Config.loss_type
    cfg_scale: float
    timestep_sampling: str

    lambda_instruction: float

    warm_start: bool
    gamma_anchor: float

    num_steps: int
    learning_rate: float

    clip_model: str

    cache_dir: str
    use_cache: bool

    seed: int
    device: str
    dtype: str


@dataclass
class LocalBlendConfig:
    """LocalBlend mask-gating config (configs/local_blend.yaml).

    See docs/p2p_pnp/local_blend specification (or RESEARCH_PROPOSAL.md
    Appendix A).
    """

    enabled: bool
    threshold: float
    base_resolution: int
    dilate_iters: int


@dataclass
class DDIMConfig:
    """Inner DDIM + null-text-optimization sub-config of EditConfig."""

    num_steps: int
    cfg_scale: float
    null_text: dict  # {"enabled": bool, "opt_steps": int, "lr": float}


@dataclass
class CrossAttentionConfig:
    """Inner cross-attention sub-config of EditConfig."""

    cross_replace_steps: float
    layer_indices: list | None


@dataclass
class EditConfig:
    """End-to-end editing pipeline config (configs/edit.yaml).

    P2P-only by design (PnP self-attention injection was scoped out;
    see RESEARCH_PROPOSAL.md and docs/p2p_pnp/REFERENCES.md).
    """

    sd_model: str
    ddim: DDIMConfig
    cross_attention: CrossAttentionConfig
    alignment_method: str        # "cosine_threshold" only in v1
    alignment_threshold: float   # τ in proposal §3.3 (cosine cutoff)
    mode: str                    # "replace" only in v1
                                 # "add" / "explicit_replace" → NotImplementedError
                                 # See RESEARCH_PROPOSAL.md §3.0.
    device: str
    dtype: str


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _REPO_ROOT / "configs"


def _load_yaml(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = _CONFIG_DIR / path
    with open(path) as f:
        return yaml.safe_load(f)


def load_pez_1(path: Path | str = "pez_1.yaml") -> Pez1Config:
    return Pez1Config(**_load_yaml(path))


def load_pez_2(path: Path | str = "pez_2.yaml") -> Pez2Config:
    return Pez2Config(**_load_yaml(path))


def load_local_blend(path: Path | str = "local_blend.yaml") -> LocalBlendConfig:
    return LocalBlendConfig(**_load_yaml(path))


def load_edit(path: Path | str = "edit.yaml") -> EditConfig:
    raw = _load_yaml(path)
    raw["ddim"] = DDIMConfig(**raw["ddim"])
    raw["cross_attention"] = CrossAttentionConfig(**raw["cross_attention"])
    return EditConfig(**raw)
