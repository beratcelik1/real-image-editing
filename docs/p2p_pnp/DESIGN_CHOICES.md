# P2P / PnP — Design Choices

Open decisions encountered while implementing the attention-editing
side of the project (existing P2P/PnP controllers, LocalBlend
additions, semantic-alignment fallback).

When implementing code in this aspect and facing an ambiguous design
decision the research proposal does not pin down, add a numbered
section below: state the question, lay out the options with
trade-offs, and (when chosen) record the decision and rationale.

See [`attention_control/DESIGN_CHOICES.md`](../../attention_control/DESIGN_CHOICES.md)
for the format precedent.

---

## 1. LocalBlend integration with existing controllers

How does the LocalBlend mask reach `CrossAttentionController` and
`SelfAttentionController` so it can gate `_word_swap` and `_inject`?

| Option | Trade-off |
|---|---|
| Add an optional `local_blend` parameter to each controller's `__init__` | Minimal API surface change; backward-compatible; `None` default preserves existing behavior |
| Pass mask as a per-call argument | Pollutes the controller `__call__` signature; each layer's processor would need to forward it |
| Globally-stored singleton | Decouples too much; hard to test |
| Dependency injection via processor | Even more boilerplate |

**Decision (commit 1):** option 1 — optional `local_blend=None` parameter
on both controllers' `__init__`. When `None`, controllers behave
exactly as before (existing tests / notebooks unaffected). When set,
the controllers (a) call `local_blend.record_cross_attention(...)` from
the cross-attention controller and (b) consult `local_blend.get_mask(spatial)`
in `_word_swap` / `_inject` to spatially blend source/target attention.
Both controllers call `local_blend.step()` from their own `step()`;
`LocalBlend.step()` is responsible for being idempotent within a single
denoising step (the implementation is in commit 3).

**Why two-controller integration vs. one:** P2P (cross-attention) and
PnP (self-attention) can both benefit from spatial gating — additive
edits need new content to have both lexical attention room (cross) and
patch-binding room (self). Sharing one `LocalBlend` instance across
both ensures the mask is consistent.

---

## 2. LocalBlend mask aggregation strategy

How does the mask get computed from per-layer cross-attention maps
that come at multiple spatial resolutions (8x8, 16x16, 32x32, 64x64
for SD2.1)?

| Option | Trade-off |
|---|---|
| Aggregate at the highest resolution | Most detail; expensive memory; smaller layers contribute less signal |
| Aggregate at the lowest resolution | Fast; loses detail; smaller layers' weak signal dominates |
| Aggregate at one canonical resolution (e.g., 16x16), resample inputs | Balance of detail and speed; canonical resolution is a tunable hyperparameter |
| Per-resolution masks, separate per-layer | Complex; doesn't share computation |

**Decision (commit 3):** option 3, with `base_resolution = 16` as the
default (configurable in `configs/local_blend.yaml`). Each cross-
attention layer's contribution is interpolated to 16x16 and averaged
into a single accumulator. At consumption time
(`get_mask(spatial)`), the canonical mask is resampled to whatever
resolution the consuming layer needs.

16x16 is chosen because:
- It's a midpoint of SD2.1's attention layer resolutions.
- It's small enough that aggregating across all layers is cheap.
- It's large enough to capture meaningful spatial structure.

---

## 3. Mask one-step lag

Cross-attention layers run AFTER self-attention within each
transformer block. So at denoising step `t`, the cross-attention
controller can record the cross-attention but the self-attention
controller has *already* run for step `t`.

| Option | Trade-off |
|---|---|
| Use mask from step `t-1` for step `t` injection | One step of lag; mask is "stale" by one step but inversion-trajectory-stable |
| Compute step-`t` mask in-place during step `t` | Requires running cross-attention before self-attention (changing U-Net forward order) — invasive |
| Skip self-attention gating, only gate cross-attention | Loses the structural-room-for-new-content benefit |

**Decision (commit 3):** option 1. The mask used at step `t` comes
from cross-attention recorded at step `t-1`. `LocalBlend.step()`,
called at the end of each denoising step, finalizes the accumulator
into a mask stored as `_current_mask` for the next step's
consumption. Step 0 has no mask (`get_mask` returns `None`); both
controllers gracefully fall back to unmasked behavior at step 0.

This matches the original P2P paper's local-blend lag handling.

---

## 4. Idempotent step()

Both `CrossAttentionController.step()` and
`SelfAttentionController.step()` call `LocalBlend.step()` at the end
of their per-step bookkeeping. If both controllers are registered
(typical for combined P2P+PnP), `step()` would be called twice per
denoising step.

**Decision (commit 3):** `LocalBlend.step()` is guarded by
`_finalized_step_id` — the second call within the same denoising
step is a no-op. This avoids requiring callers to coordinate.

---
