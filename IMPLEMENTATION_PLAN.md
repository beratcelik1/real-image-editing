# Implementation Plan: Basic Pipeline

This plan one-shots the **basic editing pipeline** specified in
[RESEARCH_PROPOSAL.md](RESEARCH_PROPOSAL.md): PEZ-1 → PEZ-2 →
LocalBlend → end-to-end `edit_image()`. After executing this plan, you
should be able to take a real image + instruction and produce an
edited image.

**Out of scope** (separate plans when ready):
- Bounded continuous refinement (`||Δ|| ≤ ε` perturbations)
- Evaluation metrics and ablation infrastructure
- Notebooks for R2/R4 sweeps

Read the [research proposal](RESEARCH_PROPOSAL.md) for motivation and
the architectural arguments. This plan assumes you understand:
- Section 3.1 (PEZ-1)
- Section 3.2 (PEZ-2)
- Section 3.3 (P2P integration)
- Appendix A (LocalBlend)

---

## 0. How to use this plan

This plan is meant to be **executable end-to-end by a Claude agent**.
Each step has:
- **What to build**: file path + function/class signatures
- **What to read first**: existing repo code or external references
- **Verification**: a concrete check that the step works before
  moving on
- **Configs read**: which YAML config files the step's code consumes

When you encounter an ambiguous design decision the plan does not
pin down, **document it in `docs/<aspect>/DESIGN_CHOICES.md`**
(format precedent:
[`attention_control/DESIGN_CHOICES.md`](attention_control/DESIGN_CHOICES.md)).
Don't silently pick — record the decision and rationale.

Configs are pre-populated in [`configs/`](configs/). All hyperparameters
are exposed there, not hardcoded. The user tunes experiments by
editing YAML; you implement code that reads YAML.

---

## 1. Setup

**1.1 — Initialize the PEZ submodule** (if not already done):

```bash
git submodule update --init --recursive
```

This populates [`external/pez/`](external/pez/) with Wen et al.'s
public PEZ codebase. We adapt — not vendor — their code.

**1.2 — Verify dependencies**:

```bash
pip install -r requirements.txt
pip install -r external/pez/requirements.txt
pip install pyyaml  # for config loading, if not already present
```

**1.3 — Verify existing pipeline works**:

```bash
python -m src.pipeline data/cat.jpg "a photo of a cat"
```

Should produce `outputs/cat_reconstruction.png`. Confirms DDIM
inversion + null-text optimization + reconstruction is functional
before we layer PEZ on top.

**Verification:** all three commands above succeed; existing
reconstruction PSNR ≥ 30 dB on `data/cat.jpg`.

---

## 2. Config loading utility

**File to create:** `src/config.py`

**What it does:** loads YAML configs into typed dataclasses. Used by
every other module so configs are validated at load time, not at
attribute access.

```python
# src/config.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class Pez1Config:
    prompt_length: int
    num_steps: int
    learning_rate: float
    weight_decay: float
    batch_size: int
    clip_model: str
    cache_dir: str
    use_cache: bool
    seed: int
    device: str
    dtype: str

@dataclass
class Pez2Config:
    warm_start: bool
    lambda_instruction: float
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
    enabled: bool
    threshold: float
    base_resolution: int
    dilate_iters: int

@dataclass
class EditConfig:
    sd_model: str
    ddim: dict           # nested; could be a sub-dataclass if preferred
    cross_attention: dict
    alignment_method: str
    device: str
    dtype: str

def load_config(path: str | Path, schema):
    with open(path) as f:
        data = yaml.safe_load(f)
    return schema(**data)

# Convenience loaders:
def load_pez_1():    return load_config("configs/pez_1.yaml", Pez1Config)
def load_pez_2():    return load_config("configs/pez_2.yaml", Pez2Config)
def load_local_blend(): return load_config("configs/local_blend.yaml", LocalBlendConfig)
def load_edit():     return load_config("configs/edit.yaml", EditConfig)
```

**Verification:** `python -c "from src.config import load_pez_1;
print(load_pez_1())"` prints a populated `Pez1Config`.

---

## 3. PEZ-1: reconstruction-aware source-image inversion

**Files to create:**
- `src/pez/__init__.py` (empty or re-exports)
- `src/pez/search.py`
- `src/pez/losses.py`
- `src/pez/source_inversion.py`

### 3.1 Two PEZ loss formulations

PEZ-1 supports two loss types, switchable via config (`loss_type` in
`configs/pez_1.yaml`):

**Loss A — `clip` (legacy, vanilla PEZ).** Original Wen et al.
formulation: prompt-image CLIP cosine similarity. Used as a warm-start
or fallback. Single CLIP-text-encoder pass per step.

```
L_clip = -cos_sim(clip_text_encoder(prompt).pooled, image_clip_emb)
```

**Loss B — `sds` (proposed, reconstruction-aware).** CFG-aware
score-distillation surrogate of "prompt + null-text reconstruct the
image under our pipeline." This is the loss the project's main
contribution depends on (see RESEARCH_PROPOSAL.md §3.1 for motivation).

```
t        ~ Uniform({1, ..., T})           # random diffusion timestep
eps      ~ N(0, I)                         # gaussian noise
x_t      = sqrt(alpha_t) * image_latent + sqrt(1 - alpha_t) * eps

eps_uncond = unet(x_t, t, null_text_emb)            # given null-text
eps_cond   = unet(x_t, t, prompt_emb)               # given the prompt
eps_cfg    = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

L_sds      = MSE(eps_cfg, eps)
```

Two U-Net forward+backward passes per PEZ step (uncond + cond). The
`null_text_emb` is an input — pre-computed externally.

The CFG-aware form lets the prompt focus on what null-text doesn't
already capture (geometry, layout, lighting). When null-text already
explains a property, the gradient on the prompt for that property is
small. Token-allocation efficiency improves vs. vanilla PEZ.

**TBD — null-text bootstrap.** `null_text_emb` requires a prompt to
compute (via existing `src/inversion.py:null_text_optimization`), but
the prompt is what we're optimizing. The chicken-and-egg has multiple
possible solutions (vanilla-PEZ bootstrap, generic null-text initializer,
alternating optimization, etc.); the choice has real implications for
runtime and quality and is **deferred from this plan** for separate
discussion.

For implementation: code `pez_search` to accept `null_text_emb` as an
input parameter. The caller (`pez_invert_source`) is responsible for
producing it. The bootstrap strategy is a one-line change in the
caller once we pick one.

### 3.2 Adapt the PEZ algorithm

**Read first:** [`external/pez/optim_utils.py`](external/pez/optim_utils.py)
— Wen et al.'s reference implementation. We adapt rather than
vendor; `external/pez/` stays untouched as a pinned reference.

**File:** `src/pez/losses.py`

```python
def clip_similarity_loss(
    soft_prompt: torch.Tensor,                # [N, 768] learnable
    target_image_embedding: torch.Tensor,     # [1, 768] CLIP image emb
    clip_text_encoder,
    tokenizer,
) -> torch.Tensor:
    """Loss A — vanilla PEZ. Returns scalar loss for backward."""

def sds_cfg_loss(
    soft_prompt: torch.Tensor,                # [N, 768] learnable
    image_latent: torch.Tensor,               # VAE-encoded source image
    null_text_embedding: torch.Tensor,        # [1, 77, 768] pre-computed
    cfg_scale: float,
    unet,
    scheduler,
    clip_text_encoder,                        # used for prompt encoding
) -> torch.Tensor:
    """Loss B — CFG-aware SDS with null-text. Returns scalar loss
    for backward.

    Implementation notes:
    - Sample t uniformly (or with importance sampling — see config
      knob `timestep_sampling`).
    - Use scheduler.alphas_cumprod[t] to compute alpha_t.
    - Both unet calls run with their respective conditioning.
    - The classifier-free guidance combination produces eps_cfg.
    - MSE against the sampled eps gives the surrogate loss.
    - Backprop flows through the prompt (via clip_text_encoder ->
      unet's cross-attention K/V) but NOT through unet's weights.
    """
```

**File:** `src/pez/search.py`

```python
def pez_search(
    loss_fn,                                  # callable taking soft_prompt
    prompt_length: int,
    num_steps: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
    initial_soft_prompt: torch.Tensor | None = None,  # warm-start
    projection_every: int = 1,                # project to vocab every k steps
    clip_text_encoder = None,                 # for projection step
    tokenizer = None,                         # for vocab embedding lookup
) -> tuple[list[int], torch.Tensor]:
    """Core PEZ optimization loop, generic over the loss function.

    Returns:
        - discrete token IDs (length prompt_length)
        - final soft prompt (for warm-starting other PEZ runs)

    Algorithm (per Wen et al. 2023, modified for arbitrary loss):
      1. Initialize soft prompt (random or warm-start).
      2. For num_steps:
         a. Project soft prompt to nearest CLIP vocab embeddings (hard
            projection through straight-through estimator).
         b. Compute loss = loss_fn(soft_prompt) — pluggable.
         c. Backprop. Gradient flows through projection as identity.
         d. AdamW step on the soft prompt.
      3. Return final discrete token IDs and final soft prompt.

    The loss_fn closure captures whatever inputs the chosen loss
    needs (target image embedding for L_clip; image latent + null-text
    + unet for L_sds).
    """
```

### 3.3 Source-image inversion wrapper (alternating R=2)

**File:** `src/pez/source_inversion.py`

The source-inversion wrapper implements the alternating-optimization
pipeline from RESEARCH_PROPOSAL.md §3.1. It composes three existing
or near-existing pieces: vanilla PEZ, null-text optimization (existing
in `src/inversion.py`), and SDS-PEZ-with-frozen-null-text.

```python
def pez_invert_source(
    image: PIL.Image.Image,
    config: Pez1Config,
    num_rounds: int = 2,        # R from §3.1; default R=2
) -> tuple[list[int], list[torch.Tensor]]:
    """Alternating-optimization pipeline producing both PEZ-1's
    discrete prompt AND the per-timestep null-text embeddings.

    Returns:
        - prompt_token_ids: list of N CLIP vocabulary token IDs
        - null_text_per_timestep: list of T tensors, one per denoising
          timestep, each shape [1, 77, 768]

    Algorithm (alternating with R=2 by default):
      # Round 0: bootstrap (CLIP-only, fast)
      c_0 = pez_search(loss_fn=clip_loss, ...)            # vanilla PEZ

      # Round 1
      z_T, traj = ddim_inversion(image, c_0_emb)
      N_0 = null_text_optimization(traj, c_0_emb)         # existing code
      c_1 = pez_search(loss_fn=sds_cfg_loss, null_text=N_0,
                       initial_soft_prompt=c_0_emb)

      # Round 2 (only if num_rounds >= 2)
      _, traj = ddim_inversion(image, c_1_emb)
      N_1 = null_text_optimization(traj, c_1_emb)
      c_2 = pez_search(loss_fn=sds_cfg_loss, null_text=N_1,
                       initial_soft_prompt=c_1_emb)

      return c_R, N_(R-1)  # the last refined prompt and matching null-text

    Caching:
      - Cache the final (prompt, null_text) tuple per image hash.
      - Cache intermediate (c_r, N_r) so resuming after a crash is cheap.
    """
```

**Configs read:** [`configs/pez_1.yaml`](configs/pez_1.yaml) →
`Pez1Config`. Key fields:
- `loss_type`: `"sds"` (project default) — alternating uses SDS for
  the refinement steps. `"clip"` is the bootstrap step's loss; not
  configurable.
- `cfg_scale`: 7.5 — used in both null-text optimization and SDS-PEZ.
- `num_rounds`: 2 — sets `R`. R1 ablates `R ∈ {1, 2, 3}`.
- All other fields (prompt_length, num_steps, learning_rate, etc.):
  defaults from Wen et al.

**Verification (after R=2 completes):**

```python
from src.pez.source_inversion import pez_invert_source
from src.config import load_pez_1

token_ids, null_text = pez_invert_source(
    Image.open("data/cat.jpg"), load_pez_1(), num_rounds=2,
)

# 1) Prompt sanity check — should be readable vocabulary tokens
from transformers import CLIPTokenizer
tok = CLIPTokenizer.from_pretrained(load_pez_1().clip_model)
print(tok.decode(token_ids))
# Should print a sensible description, e.g.:
# "a photograph of a black cat sitting on a couch"

# 2) Reconstruction fidelity — should be high
psnr = measure_inversion_fidelity(image, token_ids, null_text)
# Target: PSNR ≥ 28 dB on natural photos.

# 3) Geometry-partition diagnostic (per RESEARCH_PROPOSAL.md §3.1)
img_from_null_alone = generate_with_cfg_1(z_T, null_text, prompt=None)
# Save and inspect: should show source layout/lighting but generic identity.
```

If the geometry-partition diagnostic shows null-text reconstructing
the source faithfully (including identity), null-text has collapsed
to absorb everything. Investigate:
- Is the SDS-PEZ refinement step actually running (and not no-op)?
- Is the null-text optimization regularized? (existing
  `null_text_optimization` doesn't have a magnitude regularizer; if
  collapse becomes a problem, add one in `src/inversion.py`.)
- Does R=1 (skip the second round) produce a better partition?
  Test by passing `num_rounds=1`.

**On compute:**
- Vanilla PEZ: ~5 min on A100 (CLIP-only)
- Null-text optimization × 2: ~10 min total
- SDS-PEZ × 2: ~30 min total
- **Per source image: ~45 min total**, with all intermediate
  artifacts cached for resume-on-crash.

---

## 4. PEZ-2: instruction-conditioned target generation

**File to create:** `src/pez/instruction_conditioned.py`

**Read first:** Section 3.2 of the research proposal.

PEZ-2 reuses the same loss-pluggable `pez_search` from Step 3, but
with a three-term loss (source-preservation + instruction-following +
warm-start anchor). The source-preservation term mirrors PEZ-1's
loss formulation — i.e., it's also SDS-CFG-with-null-text by default.

```python
def pez_invert_with_instruction(
    image: PIL.Image.Image,
    instruction: str,
    pez_1_token_ids: list[int],
    config: Pez2Config,
    null_text_embedding: torch.Tensor | None = None,
    # ^ required when config.source_loss_type == "sds"
) -> list[int]:
    """Run PEZ-2 with three-term joint loss, warm-started from PEZ-1.

    Loss (computed at each gradient step):

      L = L_source                                  # source preservation
          + config.lambda_instruction * L_instr     # instruction following
          + config.gamma_anchor * L_anchor          # warm-start anchor

    Where:
      L_source: same as PEZ-1's loss (clip or sds-cfg, per
                config.source_loss_type). When sds, uses the same
                null_text_embedding.
      L_instr:  -cos_sim(clip_text_encoder(soft_prompt).pooled,
                          clip_text_encoder(instruction).pooled)
                Stays as text-text CLIP cosine — instruction is text,
                not image; SDS doesn't apply.
      L_anchor: ||soft_prompt - soft_prompt_init||²
                Pulls the prompt back toward PEZ-1's tokens unless
                instruction pressure overrides.

    Implementation:
      - If config.warm_start, initialize soft_prompt from
        clip_vocab_embeddings[pez_1_token_ids].
      - soft_prompt_init = soft_prompt.detach().clone()
      - Build loss_fn(soft_prompt) closure that returns
        L_source + lambda * L_instr + gamma * L_anchor.
      - Pass to pez_search().
    """
```

**Configs read:** [`configs/pez_2.yaml`](configs/pez_2.yaml) →
`Pez2Config`. Key fields:
- `source_loss_type`: `"sds"` (default) or `"clip"` — matches PEZ-1's
  loss type. The instruction-similarity term stays text-text CLIP
  regardless.
- `cfg_scale`: only used when `source_loss_type == "sds"`
- `lambda_instruction`, `gamma_anchor`, `warm_start`: as before
- `timestep_sampling`: same as PEZ-1

**Caching:** keyed on `(image_hash, instruction, pez_1_token_ids,
null_text_hash, config)`. Same pattern as PEZ-1.

**Verification:**

```python
target_ids = pez_invert_with_instruction(
    Image.open("data/cat.jpg"),
    "change the animal into a dog",
    pez_1_source_ids,
    load_pez_2(),
    clip_model, tokenizer,
)
print(tok.decode(target_ids))
# Should print something close to PEZ-1's prompt but with "cat" → "dog":
# "a photo of a dog sitting...".

# Token preservation: PEZ-1 vs. PEZ-2 token-ID overlap.
shared = sum(s == t for s, t in zip(pez_1_source_ids, target_ids))
print(f"Preservation: {shared}/{len(pez_1_source_ids)}")
# Should be high: at least 70% of positions identical (sub-claim 2).
```

If preservation is < 70% even with the warm-start anchor, increase
`gamma_anchor` in `configs/pez_2.yaml` and re-run. If `gamma_anchor`
has to go very high before preservation kicks in, that's evidence of
an implementation bug or unfortunate optimization landscape — log
findings in [`docs/pez_conditional/DESIGN_CHOICES.md`](docs/pez_conditional/DESIGN_CHOICES.md).

---

## 5. LocalBlend

**File to create:** `attention_control/local_blend.py`

**Read first:**
- [Appendix A of the research proposal](RESEARCH_PROPOSAL.md#appendix-a--localblend-specification)
- [`docs/p2p_pnp/DESIGN_CHOICES.md`](docs/p2p_pnp/DESIGN_CHOICES.md)
  for any decisions already recorded
- The existing controller in
  [`attention_control/cross_attention.py`](attention_control/cross_attention.py)
  to understand the integration points

```python
class LocalBlend:
    """Mask state for P2P cross-attention injection gating.

    Build it from the unmapped target token positions; the
    CrossAttentionController consults it to decide where to inject
    vs. where to leave the target alone so new content can render.
    """

    def __init__(
        self,
        target_token_indices: list[int],
        config: LocalBlendConfig,
    ): ...

    def record_cross_attention(
        self, attn_4d: torch.Tensor   # [batch, heads, spatial, tokens]
    ) -> None: ...

    def step(self) -> None:
        """Idempotent (forward-compat reserve): _finalized_step_id
        guard makes this safe under any caller pattern."""

    def get_mask(self, spatial_resolution: int) -> torch.Tensor | None: ...

    def reset(self) -> None: ...
```

Then **modify** `CrossAttentionController` to optionally accept a
`local_blend: LocalBlend | None` parameter:

- `CrossAttentionController.__call__`: call
  `local_blend.record_cross_attention(attn_weights)` after computing
  attention but before P2P injection. Modify `_word_swap` to gate the
  column copy by the mask (see Appendix A code sketch in the proposal).
- `CrossAttentionController.step()` calls `local_blend.step()` at the
  end of each denoising step to advance the mask state.

**Configs read:** [`configs/local_blend.yaml`](configs/local_blend.yaml) →
`LocalBlendConfig`.

**Verification:** standalone test before wiring into the full pipeline:

```python
# Manually invert a real image, set up controllers WITH local_blend,
# run an additive edit ("a photo of a dog" → "a photo of a dog with
# a bowtie"), and compare to the same edit WITHOUT local_blend.

# Expect: with local_blend, the bowtie renders cleanly on the dog's
# neck while the dog's pose is preserved. Without, the bowtie is
# faint or missing.
```

---

## 6. Token alignment

**File to create:** `src/splice/align.py` (and `src/splice/__init__.py`)

**Note:** LCS alignment over token IDs is already in
[`attention_control/cross_attention.py:64-80`](attention_control/cross_attention.py#L64-L80)
as `compute_token_mapping`. We add a thin wrapper that returns both
the mapping and the unmapped target indices (which LocalBlend needs).

```python
def align_pez_prompts(
    source_token_ids: list[int],
    target_token_ids: list[int],
    method: Literal["lcs", "semantic"] = "lcs",
) -> tuple[dict[int, int], list[int]]:
    """Align two PEZ-derived prompts.

    Returns:
        - mapping: {source_pos: target_pos} for matched tokens
        - unmapped_target_indices: target positions with no source
          match (used as LocalBlend's target_token_indices)

    method="lcs" calls existing compute_token_mapping(); fast and
    sufficient for warm-started PEZ-2 outputs (most positions match
    by construction).

    method="semantic" is the fallback for noisy alignment cases
    (DEFER: not in this plan; see Appendix B in the research proposal).
    """
```

**Verification:** sanity check on identity prompts:

```python
mapping, unmapped = align_pez_prompts(ids, ids)
# mapping should be {0:0, 1:1, ..., N-1:N-1}; unmapped should be [].
```

And on a known substitution:

```python
src = tok.encode("a photo of a cat sitting")
tgt = tok.encode("a photo of a dog sitting")
mapping, unmapped = align_pez_prompts(src, tgt)
# All positions except the cat/dog one should be in mapping;
# the cat/dog target position should be in unmapped.
```

---

## 7. End-to-end pipeline

**File to modify/extend:** `src/pipeline.py`

Add a new top-level function `edit_image()` that wires everything
together. Keep the existing `invert_and_reconstruct()` for the
baseline reconstruction tests; don't remove it.

The pipeline is structured around the **two-knob ablation framework**
described in RESEARCH_PROPOSAL.md §4.5:

- **Knob 1** = PEZ-2 hyperparameters (`λ_instruction`, `γ_anchor`).
  Determines how far the target prompt diverges from PEZ-1.
  Controlled via `configs/pez_2.yaml`.
- **Knob 2** = editing-time hyperparameters (`cross_replace_steps`,
  `self_replace_steps`). Determines how aggressively the target
  prompt drives rendering vs. P2P preserving source structure.
  Controlled via `configs/edit.yaml`.

```python
def edit_image(
    image_path: str,
    instruction: str,
    output_dir: str = "outputs",
    pez_2_overrides: dict | None = None,    # Knob 1 overrides
    edit_overrides: dict | None = None,     # Knob 2 overrides
) -> PIL.Image.Image:
    """End-to-end: real image + natural-language instruction → edit.

    Pipeline (mirrors RESEARCH_PROPOSAL.md §3.7):
      1. Load image.
      2. PEZ-1: pez_invert_source(image) → (source_token_ids, null_text)
                Cached per image; expensive (~70 min on first run).
      3. PEZ-2: pez_invert_with_instruction(
                  image, instruction, source_token_ids, null_text,
                  config=load_pez_2() with pez_2_overrides applied
                ) → target_token_ids
                ↑ KNOB 1: pez_2_overrides controls (λ, γ)
      4. Tokenize source/target prompts back to strings; encode through
         CLIP to get encoder_hidden_states for both.
      5. DDIM-invert source under source encoding → z_T (existing
         src/inversion.py).
      6. align_pez_prompts(source_token_ids, target_token_ids)
         → mapping, unmapped_target
      7. Set up CrossAttentionController with local_blend, configured
         by load_edit() with edit_overrides applied. Register the
         controller's processor on the U-Net's cross-attention layers.
                ↑ KNOB 2: edit_overrides controls cross_replace_steps
      8. Run editing denoising loop with [source, target] batch
         using the source's z_T as the starting noise.
      9. Decode, save, return.
    """
```

**Configs read:** all four — `pez_1`, `pez_2`, `local_blend`, `edit`.

The `pez_2_overrides` and `edit_overrides` parameters let the caller
sweep over Knob 1 and Knob 2 without touching YAML files. Useful for
running the 2D ablation grid (RESEARCH_PROPOSAL.md §4.5).

**Two-knob ablation script.** A separate utility that runs the 2D
sweep:

```python
# scripts/sweep_2d.py
from itertools import product
from src.pipeline import edit_image

KNOB_1_SETTINGS = {
    "conservative": {"lambda_instruction": 0.5, "gamma_anchor": 1.0},
    "moderate":     {"lambda_instruction": 1.0, "gamma_anchor": 0.1},
    "aggressive":   {"lambda_instruction": 3.0, "gamma_anchor": 0.01},
}
KNOB_2_SETTINGS = {
    "subtle":   {"cross_attention.cross_replace_steps": 0.8},
    "moderate": {"cross_attention.cross_replace_steps": 0.5},
    "loud":     {"cross_attention.cross_replace_steps": 0.3},
}

for k1_name, k1 in KNOB_1_SETTINGS.items():
    for k2_name, k2 in KNOB_2_SETTINGS.items():
        out = edit_image(
            image_path="data/test/dog.jpg",
            instruction="change the dog into a cat",
            output_dir=f"outputs/sweep/{k1_name}/{k2_name}",
            pez_2_overrides=k1,
            edit_overrides=k2,
        )
```

**Cost-aware ordering.** PEZ-2 is the inner-expensive operation (~10
min per Knob-1 setting). Editing is cheap (~2 min per Knob-2
setting). Iterate over Knob-1 in the outer loop and Knob-2 in the
inner loop:

```python
for k1 in KNOB_1_SETTINGS:
    target_token_ids = pez_invert_with_instruction(... k1 ...)  # ~10 min
    for k2 in KNOB_2_SETTINGS:
        edit = run_editing(... target_token_ids ..., k2 ...)    # ~2 min
        save(edit, k1, k2)
```

For 3×3 grid: 3 × 10 min PEZ-2 + 9 × 2 min edit = ~48 min per (image,
instruction) pair. Massively cheaper than 9 × (10 + 2) min = 108 min
naive.

**Verification:**

```bash
python -c "from src.pipeline import edit_image; \
  edit_image('data/cat.jpg', 'change the cat into a dog')"
```

Output should be saved to `outputs/cat_edited.png` and visually:
- Same composition, pose, background as the source cat
- A dog where the cat was

If structural preservation fails (totally different pose/composition),
the issue is likely in the controller wiring or LocalBlend setup —
check that `register_combined_control()` ran and that
`mapping`/`unmapped_target` look sensible.

If the dog doesn't render (or stays cat-like), the issue is likely
in PEZ-2 — check that `target_token_ids` actually contain "dog" or
similar by decoding them to a string.

---

## 8. Smoke-test suite

**File to create:** `tests/test_smoke.py`

Light end-to-end tests that catch obvious regressions:

```python
def test_config_loading():
    """All four configs load and produce populated dataclasses."""
    assert load_pez_1().prompt_length > 0
    assert load_pez_2().lambda_instruction >= 0
    assert load_local_blend().threshold > 0
    assert load_edit().sd_model

def test_pez1_returns_valid_token_ids():
    """PEZ-1 on a small test image returns the expected number of
    token IDs, all within CLIP's vocab range."""

def test_pez2_warm_start_preserves_majority():
    """PEZ-2 with high gamma_anchor preserves >50% of source tokens
    even with a strong instruction (sanity check on warm-start)."""

def test_align_identity():
    """Aligning identical prompts gives identity mapping, empty
    unmapped."""

def test_local_blend_step_is_idempotent():
    """Calling LocalBlend.step() twice in the same step doesn't
    double-finalize."""

def test_edit_image_runs_without_error():
    """End-to-end on a tiny test image produces output. Don't assert
    quality — just that it runs."""
```

These should all pass before considering the basic pipeline complete.

---

## 9. What to do if something doesn't work

**PEZ-1 reconstruction worse than BLIP-2 baseline:**
- Increase `prompt_length` (more detail capacity)
- Increase `num_steps` (more optimization)
- Verify CLIP model matches SD's CLIP variant exactly
- Log findings in [`docs/pez_conditional/DESIGN_CHOICES.md`](docs/pez_conditional/DESIGN_CHOICES.md)

**PEZ-2 token preservation rate too low:**
- Increase `gamma_anchor` (stronger warm-start pull)
- Decrease `lambda_instruction` (weaker instruction pull)
- Verify the warm-start initialization is actually being used (log the
  initial soft prompt vs. PEZ-1's vocab embeddings)

**P2P edit produces no change vs. source:**
- Verify the [source, target] batch is correctly stacked
- Verify `mapping` has reasonable matched positions
- Verify the cross-attention controller is actually registered on the
  U-Net's cross-attention layers

**Edit destroys structure (dog appears but pose changes):**
- Increase `cross_replace_steps` (more P2P injection — keeps source
  attention longer in the denoising trajectory)
- Verify LocalBlend mask isn't covering the entire image (log
  `mask.sum() / mask.numel()` per step)

**LocalBlend mask is bad (covers wrong region or whole image):**
- Verify `target_token_indices` contains the right positions
- Adjust `threshold` in `configs/local_blend.yaml`
- Check the mask resolution matches what each layer expects

---

## 10. After this plan

Once the basic pipeline works end-to-end, the next plans (separate
documents, not yet written) cover:

- **Refinement plan**: bounded continuous Δ refinement on PEZ-1 and
  PEZ-2 outputs (`||Δ|| ≤ ε`), characterizing the fidelity vs.
  position-stability Pareto frontier (R3 in the proposal).
- **Evaluation plan**: implementing footprint concentration, edit
  quality metrics, ablation infrastructure (R4 in the proposal).
- **Notebooks plan**: R1, R2, R3, R4 ablation notebooks running the
  full experimental sweep.

Each layers cleanly on top of the basic pipeline without modifying
it. The basic pipeline is the foundation; everything else is
additions.
