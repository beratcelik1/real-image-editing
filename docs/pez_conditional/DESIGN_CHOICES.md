# PEZ-Conditional — Design Choices

Open decisions encountered while implementing the PEZ side of the
project: PEZ-1 (source-image inversion), PEZ-2 (instruction-conditioned
target generation), and bounded continuous refinement.

When implementing code in this aspect and facing an ambiguous design
decision the research proposal does not pin down, add a numbered
section below: state the question, lay out the options with
trade-offs, and (when chosen) record the decision and rationale.

See [`attention_control/DESIGN_CHOICES.md`](../../attention_control/DESIGN_CHOICES.md)
for the format precedent.

---

## 1. Token alignment fallback (semantic vs LCS)

`align_pez_prompts` in `src/splice/align.py` accepts a `method`
parameter taking `"lcs"` or `"semantic"`. The semantic-alignment
fallback (CLIP-embedding-based bipartite matching) is documented as
Appendix B of the research proposal but is **not implemented in v1**.

| Option | Trade-off |
|---|---|
| Implement both LCS and semantic, default to LCS | Provides a fallback if LCS produces noisy mappings on PEZ-derived prompts |
| Implement only LCS, leave method param for forward compatibility | Smallest surface area; raises `NotImplementedError` if `method="semantic"` is requested |

**Decision (commit 1):** the `method` parameter exists in the
`align_pez_prompts` signature for forward compatibility, but
`method="semantic"` raises `NotImplementedError` pointing to Appendix
B. This keeps the API stable across v1 (LCS) and any future v2 that
adds semantic alignment, without bloating v1 with code we don't yet
need. PEZ-2's warm-start property means most positions match by
token-ID, so LCS is expected to suffice.

---

## 2. Config schema

`src/config.py` uses Python `dataclass` schemas with `pyyaml` loading.

| Option | Trade-off |
|---|---|
| Plain dataclasses + manual yaml.safe_load | Smallest dependency footprint; type errors only at attribute access |
| Pydantic models | Validation at load time; extra dependency |
| Hydra | Full experiment-config framework; heavyweight |

**Decision (commit 1):** plain dataclasses + `yaml.safe_load`. We don't
need runtime validation beyond what dataclasses provide (dataclass
constructor raises `TypeError` if YAML keys don't match fields, which
catches structural errors at load time). Avoiding a Pydantic dependency
keeps the requirements small. If we later need richer validation,
swapping the dataclass schemas for Pydantic models is mechanical.

---

## 3. Re-implementing `nn_project` rather than importing from `external/pez/`

`external/pez/optim_utils.py:nn_project` performs the straight-through
nearest-vocabulary projection that's the heart of PEZ. Its
implementation depends on `sentence_transformers.util` (for
`semantic_search`, `dot_score`, `normalize_embeddings`) — utilities
that are easy to express in raw PyTorch but bring a heavy package
dependency.

| Option | Trade-off |
|---|---|
| Import `nn_project` from `external/pez/` | Adds `sentence_transformers` (~100 MB + transitive deps) just for this one utility; would also add `open_clip` if we wanted other PEZ helpers |
| Re-implement `nn_project` in raw PyTorch | ~10 lines; no new dependencies; documents an explicit deviation from the submodule |

**Decision (commit 2):** re-implement in raw PyTorch
(`src/pez/search.py:nn_project`). The straight-through estimator's
implementation is mechanical — normalize, argmax over cosine
similarity, embed lookup, identity-gradient route. Adding
`sentence_transformers` as a dependency for one utility is overkill.

The submodule remains as a reference; future contributors can
cross-check our implementation against `external/pez/optim_utils.py`.

---

## 4. Custom CLIP encoder forward pass for `inputs_embeds`

PEZ's losses need to run a *custom embedding sequence* (the projected
soft prompt) through CLIP's text transformer, not a sequence of token
IDs. HuggingFace `CLIPTextModel.forward` expects `input_ids` and does
the embedding lookup internally; it doesn't natively support
`inputs_embeds`.

Three options:

| Option | Trade-off |
|---|---|
| Bypass `CLIPTextModel.forward` and call `text_model.encoder` directly with our embeddings | Reimplements the small wrapping glue (positional embeddings, causal mask, final layer norm, EOS pooling) but gives us full control |
| Monkey-patch `CLIPTextModel` to accept `inputs_embeds` | Fragile; depends on internal HF API |
| Wrap our embeddings as a tensor, encode token IDs, and replace the embedded tensor in-place | Brittle; relies on HF internals |

**Decision (commit 2):** option 1.
`src/pez/losses.py:encode_through_text_model` replicates the core of
`CLIPTextTransformer.forward`: position-embedding addition + causal
attention mask + encoder + final layer norm + EOS pooling. The
implementation is ~30 lines and depends only on stable HF API surface
(`text_model.embeddings.position_embedding`, `text_model.encoder`,
`text_model.final_layer_norm`).

This is the same approach taken by P+ (Voynov 2023) and several other
papers that operate on custom CLIP-text-encoder inputs.

---

## 5. CLIP image features required as input to `pez_invert_source`

The vanilla-PEZ bootstrap step (Round 0 of alternating R=2) uses
`clip_similarity_loss`, which compares the prompt's pooled CLIP
encoding against an image's CLIP image embedding. Two options for
where the image embedding comes from:

| Option | Trade-off |
|---|---|
| Load a `CLIPModel` (image+text encoder) inside `pez_invert_source` | Adds ~600 MB of model weights to the loaded state; many users will already have a CLIP image encoder loaded elsewhere |
| Require the caller to pass `clip_image_features` | Caller has flexibility (uses whichever CLIP variant they prefer, or memoizes across many PEZ runs); pez_invert_source raises ValueError if missing |

**Decision (commit 2):** option 2. `pez_invert_source` accepts
`clip_image_features` as an optional argument and raises
`ValueError` if it's missing when `loss_type == "clip"`. The caller
(typically `src/pipeline.py:edit_image`) is responsible for computing
or providing the image embedding. This decouples PEZ-1 from CLIP
loading and enables better cost amortization (one `CLIPModel` load,
many PEZ-1 runs).

---

## 6. PEZ-2 requires `sd_components` from caller (no internal load)

Unlike `pez_invert_source` (which has an internal SD-load fallback for
convenience), `pez_invert_with_instruction` requires the caller to
pass `sd_components` explicitly.

**Reason:** PEZ-2 is intended to run many times per source image
(once per instruction × Knob-1 setting). If each call loaded SD
internally, we'd reload ~6 GB of weights repeatedly. Forcing the
caller to load once and pass it in makes the cost amortization
explicit. The end-to-end `edit_image` (commit 4) handles this.

---

## 7. LCS logic inlined in `src/splice/align.py`

`align_pez_prompts` originally imported `compute_token_mapping` from
`attention_control/cross_attention.py`, which transitively imports
torch. This made `src/splice/align.py` impossible to import without
torch installed — blocking environments where we just want to run
the alignment logic in unit tests.

**Decision (commit 4):** inline a copy of the LCS logic
(`_compute_token_mapping_lcs`) directly in `src/splice/align.py`.
The function is 5 lines and uses only `difflib` from the standard
library. The original `compute_token_mapping` in
`attention_control/cross_attention.py` is unmodified for backward
compatibility with existing notebooks and code.

This is a deliberate tiny duplication: the LCS algorithm is simple
enough that maintaining two copies isn't a meaningful burden, and
the dependency-decoupling benefit (torch-free unit tests) outweighs
the duplication cost. Recorded explicitly so a future agent doesn't
"deduplicate" by re-introducing the cross-module import.

---
