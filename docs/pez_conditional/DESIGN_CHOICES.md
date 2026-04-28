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
