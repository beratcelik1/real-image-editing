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
