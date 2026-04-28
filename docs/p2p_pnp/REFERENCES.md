# P2P references and hyperparameter sources

This file documents where each P2P hyperparameter default in
[configs/edit.yaml](../../configs/edit.yaml) and
[configs/local_blend.yaml](../../configs/local_blend.yaml) comes from
in the published reference, plus citation pointers.

## Reference repo

[external/prompt-to-prompt/](../../external/prompt-to-prompt/) —
Google's official Prompt-to-Prompt implementation, submodule of
[github.com/google/prompt-to-prompt](https://github.com/google/prompt-to-prompt).

The repo is structured as research notebooks rather than an
importable library; we treat it as a reference (citation +
hyperparameter source + cross-validation against bugs) rather than as
a runtime dependency. Our `attention_control/cross_attention.py` is
adapted from their notebook code (`prompt-to-prompt_stable.ipynb`)
and follows the same algorithm.

## Citations

- **Hertz et al. 2022** — *Prompt-to-Prompt Image Editing with Cross
  Attention Control*. arXiv:2208.01626.
  Introduces the cross-attention swap, attention re-weighting, and
  prompt refinement operations that our `CrossAttentionController`
  implements. Also introduces the `LocalBlend` concept.

- **Mokady et al. CVPR 2023** — *Null-Text Inversion for Editing Real
  Images using Guided Diffusion Models*. arXiv:2211.09794.
  The null-text optimization our `src/inversion.py:null_text_optimization`
  implements. Required to make P2P work on real images under CFG > 1.

- **Wen et al. 2023** — *Hard Prompts Made Easy: Gradient-Based Discrete
  Optimization for Prompt Tuning and Discovery* (PEZ). arXiv:2302.03668.
  The discrete prompt search algorithm we adapt for PEZ-1 and PEZ-2.

## Hyperparameter sources

Each default in our YAML configs has an upstream source in Google's
notebook. When tuning, cross-reference these.

### `configs/edit.yaml`

| Field | Our default | Google's reference | Source |
|---|---|---|---|
| `ddim.num_steps` | 50 | 50 | `prompt-to-prompt_stable.ipynb` |
| `ddim.cfg_scale` | 7.5 | 7.5 | Standard CFG default |
| `ddim.null_text.opt_steps` | 10 | 10 (per timestep) | `null_text_w_ptp.ipynb` |
| `ddim.null_text.lr` | 0.01 | 0.01 (Adam) | `null_text_w_ptp.ipynb` |
| `cross_attention.cross_replace_steps` | 0.8 | varies by edit type — see below | `prompt-to-prompt_stable.ipynb` |

**`cross_replace_steps` per edit type** (from Google's notebooks):

| Edit type | Recommended `cross_replace_steps` | Rationale |
|---|---|---|
| Word swap (`cat → dog`) | 0.4 | Lower is better — gives target prompt late-step rendering room |
| Adding tokens (`+ bowtie`) | 0.8 | Higher preserves source structure while new tokens render in unmasked regions |
| Re-weighting (`amplify "cat"`) | 1.0 | Full injection — re-weighting is multiplicative, doesn't need replacement |

The 0.8 default in our config is the additive-edit setting. For
substitution edits, ablate down toward 0.4 (this is the "Knob 2"
sweep dimension in the proposal).

### `configs/local_blend.yaml`

| Field | Our default | Google's reference | Notes |
|---|---|---|---|
| `threshold` | 0.3 | 0.3 (`th=0.3`) | `prompt-to-prompt_stable.ipynb`, `LocalBlend.__init__` |
| `base_resolution` | 16 | varies by layer | Google computes per-layer; we use a canonical 16 (see `docs/p2p_pnp/DESIGN_CHOICES.md` #2) |
| `dilate_iters` | 1 | not present | Our addition — softens the binary mask boundary |

## Differences from Google's reference

We extended the original P2P implementation in two ways:

1. **`local_blend` parameter on `CrossAttentionController`**.
   Google's `LocalBlend` is a *separate object* applied externally
   to attention maps. We make it an optional parameter on the
   controller so the gating happens inside `_word_swap` directly.
   Mechanically equivalent; just a different API surface.
   See `docs/p2p_pnp/DESIGN_CHOICES.md` #1.

2. **Multi-resolution mask aggregation at canonical 16×16**.
   Google's reference handles each cross-attention layer
   independently with per-layer mask handling. We aggregate to one
   canonical resolution, then resample to whatever the consuming
   layer needs. Trades minor per-layer fidelity for simpler mask
   bookkeeping. See `docs/p2p_pnp/DESIGN_CHOICES.md` #2.

3. **One-step lag** (consequence of single-pass forward through the
   transformer): mask used at step `t` comes from cross-attention
   recorded at step `t-1`. Google's reference also has this
   property. See `docs/p2p_pnp/DESIGN_CHOICES.md` #3.

## Cross-validation procedure

If our P2P implementation produces unexpected results, cross-check
against Google's notebooks:

1. Open `external/prompt-to-prompt/prompt-to-prompt_stable.ipynb` and
   `null_text_w_ptp.ipynb` in Jupyter.
2. Run their reconstruction + edit example on a known image.
3. Compare attention map shapes, swap mechanics, and final edit
   visually against ours.
4. Differences in `cross_replace_steps` behavior, attention map
   orientation, or null-text reconstruction quality flag where our
   adaptation diverged from theirs.

This procedure is the recommended debugging path for any P2P-related
issues that surface during R1-R4 implementation.

## Why we don't use PnP

PnP (Tumanyan et al. CVPR 2023) is a separate technique — self-
attention map injection plus ResNet feature injection — for
text-driven image-to-image translation. It's structurally orthogonal
to our project's contribution (which is at the **prompt level**, where
P2P operates).

PnP's mechanism is text-agnostic: it doesn't benefit from the
prompt-level improvements PEZ-1/PEZ-2 provide. Including PnP in our
v1 pipeline would add structural-preservation strength that doesn't
showcase our contribution.

PnP is therefore deliberately excluded from the v1 codebase. If
stronger structural preservation is needed in v2, PnP can be re-added
as a configuration toggle without architectural changes (the existing
`CrossAttentionController` is independent of self-attention).
