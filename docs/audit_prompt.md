# Bug-audit prompt for fresh Claude agents

A self-contained prompt for spawning a Claude subagent to audit this
codebase for correctness bugs. To re-run the audit, ask the running
Claude session something like:

> Run the audit prompt in `docs/audit_prompt.md` against the current
> state of the repo. Use a `general-purpose` subagent. Don't let it
> make code changes — report only.

The session will translate that into an `Agent(subagent_type=
"general-purpose", prompt=…)` call with the prompt text below.

Re-run after every set of architectural changes; bug class #1
(data-flow consumption gaps) regresses every time a new artifact is
added to the pipeline.

---

## Prompt body (everything below is what the agent receives)

You are auditing a research codebase that combines:

- Continuous-prompt optimization (PEZ-style; soft prompts in CLIP-input
  embedding space — see `RESEARCH_PROPOSAL.md` §3.1).
- Stable Diffusion DDIM inversion + null-text optimization (Mokady-style).
- Score Distillation Sampling (SDS) with classifier-free guidance.
- Prompt-to-Prompt cross-attention manipulation (Hertz-style).
- Caching of expensive optimization outputs to disk.

Your job is to find **correctness bugs**, not style issues. **Report
only — do not modify any code.** The user will review your findings
and decide what to fix.

## Files in scope

```
src/pez/search.py                       (continuous-PEZ optimization loop)
src/pez/source_inversion.py             (PEZ-1 alternating R=2 wrapper)
src/pez/instruction_conditioned.py      (PEZ-2 joint loss)
src/pez/losses.py                       (SDS-CFG, CLIP-cosine losses)
src/splice/align.py                     (per-position cosine alignment)
src/inversion.py                        (DDIM inversion + null-text optim)
src/pipeline.py                         (edit_image, run_p2p_edit, loop)
attention_control/cross_attention.py    (P2P controller + processor)
attention_control/local_blend.py        (spatial mask gating)
src/config.py + configs/*.yaml          (config schema vs YAML)
notebooks/colab_oneshot.ipynb           (sweep notebook)
tests/test_smoke.py
```

`RESEARCH_PROPOSAL.md` is the design intent. `IMPLEMENTATION_PLAN.md`
is the implementation contract. **If the code disagrees with both,
that's a bug.**

## Bug classes, in priority order

1. **DATA FLOW / CONSUMPTION GAPS.** A computed expensive output that
   is passed around but never actually used at the point that needs
   it (the canonical example: null-text inversion exists but the
   editing loop uses default uncond instead). Trace each artifact
   from production to consumption.

2. **INDEX / TIMESTEP ALIGNMENT.** With multiple coordinate systems
   in play — PEZ position 0..N-1, CLIP context 0..76 with BOS at 0,
   scheduler denoising-step index 0..T-1, training-timestep value
   0..1000 — bugs creep in via off-by-ones and missed conversions.
   Specifically check: any place an index is converted from one
   system to another; any sampling that should be coupled but is
   independent; whether `null_text_per_timestep[i]` is consistently
   used at the same `i` as the scheduler step.

3. **DEVICE / DTYPE CONSISTENCY ON CACHE HITS.** Cache loads use
   `torch.load(map_location='cpu')`. Anywhere a tensor is later
   passed to a CUDA model without an explicit `.to(device=device)`,
   it crashes — but only on cache-hit paths. `.to(dtype)` does NOT
   change device.

4. **CACHE KEY COMPLETENESS.** Walk every cache hash function. List
   the config fields it includes. Cross-check against the config
   dataclass for any field that affects optimization output but
   isn't in the hash. Tuning a missing field gives silently stale
   cached results.

5. **AUTOGRAD GRAPH WASTE.** Look for U-Net or large-model forward
   passes inside the gradient loop where the gradient cannot reach
   the optimization variable. The graph is built and discarded —
   wastes ~half the GPU memory. Should be `torch.no_grad()`-wrapped.

6. **REDUNDANT WORK IN SWEEP LOOPS.** The notebook iterates over
   (λ, γ) outer × cross_replace_steps inner. Anything inside the
   inner loop that depends only on outer-loop variables is redundant
   — should be hoisted out.

7. **CFG BATCH ORDERING.** Editing loops assemble `[uncond, cond]`
   or `[uncond_src, uncond_tgt, cond_src, cond_tgt]` batches.
   Verify the `chunk()` / `split()` recovers them in the same order
   they were stacked. Easy to swap source/target halves.

8. **SCHEDULER STATE.** `scheduler.set_timesteps()` is stateful and
   shared. If function A sets it to 50 and function B (also called)
   expects 50 too, but A was last called with 30, B sees the wrong
   values. Check every `set_timesteps` call and confirm consumers
   know what state they expect.

9. **ATTENTION TENSOR SHAPES.** Cross-attention probabilities have
   shape `[batch, heads, spatial, tokens]`. Check controllers slice
   the right half (source vs target) and the right token positions,
   accounting for any BOS offset.

10. **STATE RESET.** Stateful objects (`CrossAttentionController`,
    `LocalBlend`, schedulers) — verify `reset()` is called or new
    instances are created between sweep iterations. Otherwise prior
    state contaminates next iteration.

11. **CONFIG SCHEMA DRIFT.** For each YAML in `configs/`, verify
    every key exists in the corresponding dataclass and vice versa.
    A YAML key not in the dataclass silently does nothing; a
    dataclass field not in the YAML raises `TypeError` on load.

12. **NUMERICAL EDGE CASES.** `F.cosine_similarity` has `eps=1e-8`
    default — OK. But any custom normalization, division, or sqrt
    should be checked for zero/negative input. Also fp16
    underflow in losses.

13. **EDGE CASES IN CONTROL FLOW.** `matched=[]` or `unmapped=[]`
    (all positions drifted, or none); `prompt_length=1`;
    `num_steps=0`; image dimensions that don't square
    (cross-attention spatial ≠ H × W).

14. **SEMANTIC BUGS IN DIFFUSION.** DDIM inversion direction
    (forward vs reverse), `alphas_cumprod` indexing at boundary
    (t=0 vs t=T-1), null-text optimization timestep ordering (does
    `null_embeddings[0]` correspond to the highest-noise step or
    lowest?), CFG sign (should be `ε_cfg = ε_uncond + s·(ε_cond -
    ε_uncond)`, not the inverse).

## Output format

For each bug found, report:

- **Severity:**
  - `CRITICAL` — silent wrong gradient or wrong output
  - `REAL` — crash, silent staleness, or wrong behavior on a
    common path
  - `PERF` — wasted memory or compute (correct output)
  - `STYLE` — cosmetic; mention only if it hides a real bug
- **File and line range** (e.g., `src/pez/losses.py:280-285`)
- **Description** — one to two sentences on what's wrong
- **Fix** — one to two sentences on what would resolve it
- **Effort** — estimated lines of change

Order findings by severity. Within severity, group by file so the
user can review one file at a time.

## Anti-patterns to skip

- Formatting, naming, docstring polish — unless they hide a real
  bug.
- "This could be more idiomatic" suggestions.
- Refactoring proposals that don't address a specific bug.
- Speculative bugs that depend on unrealistic input ranges.

The goal is bug discovery, not a code-review pass.

## A note on regressions

This codebase has been actively refactored. Recent commits include:
- `846efb9` — switch PEZ-1/PEZ-2 to continuous embeddings
- `7525108` — residual parameterization for PEZ-1 SDS rounds
- `e65902e` — PEZ-2 hardcodes weight_decay=0
- `5a1e5a1` — fixes for six bugs found in a prior audit

Pay extra attention to whether the most recent fixes (`5a1e5a1`)
introduced new bugs or only partially addressed the original ones.
Independent verification of fixes is a high-yield audit angle.
