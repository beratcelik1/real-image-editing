# Hard Edits Made Easy: Instruction-Conditioned Prompt-Embedding Inversion for Real-Image Editing

> Two PEZ-style continuous-prompt optimizations (Wen et al. 2023's
> "Hard Prompts Made Easy" run without the vocabulary-projection step,
> giving us continuous prompt embeddings rather than discrete tokens),
> applied to **both** the source-representation problem and the
> instruction-following problem, producing a fully-automated real-image
> editing pipeline:
>
> 1. **PEZ-1** runs on the source image → produces N continuous
>    CLIP-input embeddings `[N, 768]` that reconstruct the source.
>    Replaces BLIP-2 captioning as the source representation.
>
> 2. **PEZ-2** runs on `(source image, user instruction)` jointly →
>    produces N continuous embeddings satisfying both the source's
>    visual content and the instruction's semantic intent. Warm-started
>    from PEZ-1's embeddings so the two prompts share embedding values
>    at positions unaffected by the instruction. Instruction following
>    emerges from CLIP's existing semantic alignments via embedding-
>    space optimization — no LLM, no rule-based parser, no learned
>    task-specific model.
>
> A non-obvious advantage of this formulation: with large N (long
> prompts), PEZ recovers visual details the user wouldn't think to
> specify. For `"change the husky into a cat"`, PEZ-2 finds an
> embedding whose visual signature matches the source husky's coloring
> and fur texture — automatically — by satisfying the source-image-
> similarity term while moving toward the instruction. **The system
> doesn't just automate what a user would do; it does it better than
> the user could.**
>
> Existing P2P attention-editing machinery composes with PEZ-1 /
> PEZ-2 outputs unchanged: warm-start gives source/target prompts the
> same length and per-position correspondence by construction, so
> alignment reduces to a per-position cosine-distance check. Positions
> where the target embedding drifted from source by more than a
> threshold are the "unmapped" positions; they drive local-blend
> masking.
>
> Note on terminology: we keep the name **"PEZ"** because the
> optimization machinery (soft prompt + AdamW + CLIP/SD gradient flow)
> is Wen et al.'s. The change vs. their original is dropping the
> straight-through projection to vocabulary points — the soft prompt
> *is* the output, no longer an intermediate representation. Where the
> distinction matters we say **"continuous PEZ"** explicitly.

---

## 0. Quick start — where to begin

A fresh agent picking this up should:

1. Read sections 1 and 2 below for the framing.
2. Skim section 3 (the core method) — pay attention to 3.1 (PEZ-1
   on source), 3.2 (PEZ-2 instruction-conditioned), and 3.3 (P2P
   integration).
3. Jump to **section 5** (codebase organization) and **section 7**
   (Phase R1) to start writing code.
4. Refer to **Appendix A** at the end of this document **only when
   building Phase R4** — that phase needs the local-blend mechanism
   specified there.

The first concrete thing to build is a working PEZ implementation
(Phase R1) integrated with the existing DDIM inversion pipeline.
PEZ-2 (the instruction-conditioned variant) is a small modification
on top of PEZ-1 — same algorithm, additional loss term — so once R1
works, R2 is largely a loss-function change.

---

## 1. The gap

Two communities work on adjacent problems with non-interoperable primitives:

| Personalization | Attention editing |
|---|---|
| Goal: high-fidelity reconstruction of a specific concept | Goal: structurally faithful editing of a real or generated image |
| Output: a learned embedding in CLIP space that conditions SD on the concept | Output: a *modified* attention pattern that swaps content while preserving structure |
| Examples: Textual Inversion (Gal et al. 2022), DreamBooth (Ruiz et al. 2022), Custom Diffusion (Kumari et al. 2022), Concept Sliders (Gandikota et al. 2023) | Examples: Prompt-to-Prompt (Hertz et al. 2022), Plug-and-Play (Tumanyan et al. 2023), MasaCtrl, FreePromptEdit |
| Operates on: input side (embeddings, text encodings) | Operates on: middle of the network (attention maps inside U-Net) |

The two methods are not currently composable. **Textual inversion produces
embeddings that destroy P2P alignment**: a learned `e_S*` doesn't have an
identifiable per-position role in the prompt — it's "this image's gestalt"
collapsed into one 768-dim point. P2P's mechanism (copying attention
columns at matched token positions) needs the source and target prompts
to share *position correspondence*: position 5 in source carries the
same role as position 5 in target so their cross-attention columns can
be swapped meaningfully.

If your source prompt is `"a photograph of a S*"` and your target is
`"a photograph of a S* with a T*"`, the only positions occupying
content learned for this image are S* and T* — and their cross-attention
behavior is whatever happened to emerge during textual inversion
training. Nothing about the training procedure encourages those columns
to be:

- **Localized** — concentrated at one position rather than smeared across
  the contextual encoding
- **Stable** — behaving consistently when the surrounding prompt context
  changes
- **Swappable** — having a structure that mirrors how natural-token
  cross-attention columns behave (so P2P's column-swap operation
  produces sensible results)

This is the gap. Standard textual inversion optimizes a single
embedding for *reconstruction fidelity only*. It captures one image's
gestalt at one position; it doesn't give us a position-structured
multi-token representation, an instruction-conditioning mechanism, or
an alignment story for downstream attention editing.

### Our approach: two continuous-PEZ optimizations, source and instruction-conditioned

The personalization literature uses N=1 continuous embeddings
optimized for reconstruction. That single-vector design is part of
why TI doesn't compose with P2P — there's no per-position structure
for alignment to anchor on. **We propose: optimize N continuous
embeddings (N≈15) with three structural ingredients TI lacks: a
reconstruction-aware loss tied to the editing pipeline (CFG-aware
SDS with null-text), an instruction-conditioning pass with a warm-
start anchor, and per-position alignment for P2P composition.**

The pipeline runs **continuous PEZ** twice with different conditioning:

1. **PEZ-1: source-image inversion.** Replace BLIP-2 captioning with
   continuous-PEZ-on-source. The optimization produces a sequence of
   N continuous 768-dim embeddings that, when used as the input to
   CLIP's text encoder, make the model reconstruct the source image
   under our DDIM-inversion + null-text + denoising stack.
   Per-position structure is explicit: each of the N positions has a
   stable role, and the warm-start property (below) carries that role
   into PEZ-2.

2. **PEZ-2: instruction-conditioned target generation.** PEZ runs a
   second time with two simultaneous CLIP similarity targets:
   - the source image (preserve visual content)
   - the user's natural-language instruction text (apply edit)

   Warm-started from PEZ-1's embeddings, the optimization produces a
   target prompt that mostly matches PEZ-1 except at positions under
   strong instruction-pressure. This is the key technical piece —
   instruction following emerges from CLIP's pretrained semantic
   alignments through PEZ-2's joint loss, with **no LLM, no rule-
   based parser, no learned task-specific model.**

The two prompts (PEZ-1 → source, PEZ-2 → target) compose with existing
P2P unchanged because:

- **Same length, same per-position initialization** (warm-start) →
  position `i` in source corresponds to position `i` in target by
  construction. No alignment search needed.
- **Per-position cosine distance** identifies which positions drifted
  during PEZ-2's optimization. Positions where target stayed close to
  source are "matched" (P2P injects); positions where target drifted
  beyond a threshold are "unmapped" (P2P leaves alone, and the local
  blend mask is built from those positions' cross-attention).
- **Reconstruction-aware optimization** ensures each position's
  cross-attention behavior is calibrated to actual image
  reconstruction, not just CLIP cosine — this is what makes
  per-position cross-attention well-localized for P2P/LocalBlend.

The discrete-token-projection step from Wen et al.'s original PEZ is
**dropped throughout**: the soft prompt *is* the prompt. We retain the
"PEZ" name because the optimization machinery (AdamW on a soft prompt
with gradient flow through CLIP and SD's U-Net) is theirs; we just
keep the result continuous instead of snapping to vocabulary points.
This keeps strictly more reconstruction fidelity (no Voronoi
quantization loss) and avoids a Voronoi-jump nonlinearity in the
optimization landscape. For human-readable logging or debugging, an
optional snap-to-nearest-vocabulary step can be applied at inference
time *only* — never during optimization.

### Additive editing is the hardest case (and the primary showcase)

The smearing problem has two distinct severities depending on edit type:

**Replacement edits** (e.g., learned `e_cat` swapped with learned
`e_dog` at the same prompt position) are *partially forgiving*. Both
embeddings smear, but they smear at the same position, so P2P's
column-swap propagates similar smearing patterns from source to target.
The mismatch is small. Vanilla textual inversion produces edits that
are imperfect but usable.

**Additive edits** (e.g., `"a photo of a dog"` → `"a photo of a dog
with a [bowtie_S*]"`, where `[bowtie_S*]` is a learned concept at a new
position the source doesn't contain) have **two failure modes** that
replacement doesn't:

1. **Local-blend mask collapse.** Additive edits depend on the local
   blend mechanism (specified in Appendix A) to give the new concept
   spatial room. The mask is built from cross-attention to the new
   concept's column. If that column is smeared (the bowtie's "signal"
   leaks across the encoding), the cross-attention pattern is
   spatially diffuse. The mask either over-covers (loses structure)
   or under-covers (no room for the concept). Either way, the
   additive edit fails.

2. **P2P injection actively erases the concept.** This is the worse
   failure. When `e_bowtie` smears to positions 5, 6, 7 (the
   contextual encoding of nearby words like "dog", "with"), those
   positions in the *target* now carry partial bowtie information.
   P2P's column-swap then copies positions 5-7 from **source to
   target** — but the source has no bowtie at those positions; it
   has plain "dog", "with", etc. The injection overwrites the smeared
   bowtie content with source content that doesn't include bowtie.
   Net effect: the smearing puts bowtie information in positions that
   P2P then deliberately removes, leaving the concept underrendered.

Replacement edits don't have failure mode 2 because the source and
target both have learned content at the smearing positions; the
"erasure" cancels because both sides have similar smearing.

**Why our N continuous-embedding design avoids both modes.**

- *Failure mode 1* is mitigated by the SDS-CFG reconstruction-aware
  loss (Section 3.1). Each position's contribution to cross-attention
  is shaped by gradient flow through SD's U-Net during training, so
  positions naturally end up with localized cross-attention columns
  rather than smeared ones — that's the loss landscape rewarding
  position-localized contributions over diffuse ones.
- *Failure mode 2* is mitigated by the per-position warm-start
  anchor and per-position alignment. Source position `i` and target
  position `i` are the same row by construction; positions that
  drifted under PEZ-2's optimization are explicitly identified by
  cosine distance and marked *unmapped*. P2P does not inject onto
  unmapped positions, so new content at those positions stays put.

**This makes additive editing the empirical showcase for the
proposed architecture.** If our continuous-PEZ + warm-start + per-
position alignment design materially improves additive edit quality
vs. vanilla TI injection, the contribution is real and measurable.
If it only helps replacement edits, the contribution is marginal
because replacement was already mostly working.

## 2. Hypothesis

A fully-automated real-image editing pipeline built from two PEZ
optimizations — **PEZ-1** for source inversion and **PEZ-2** for
instruction-conditioned target generation — composes with existing
P2P attention-editing machinery without modification. Instruction
following emerges from CLIP's semantic alignments via PEZ-2's joint
loss, requiring no LLM, no rule-based parser, and no task-specific
learned model.

The hypothesis decomposes into four sub-claims. Each is independently
testable and could fail without the others failing.

### Sub-claim 1 — PEZ-1 as source representation beats captioning

Continuous PEZ run on the source image yields N continuous prompt
embeddings that, when used for DDIM inversion, reconstruct the source
image at higher fidelity than a BLIP-2 caption of the same image.

- **Measurement.** PSNR / SSIM / LPIPS on source images reconstructed
  via DDIM inversion + null-text optimization. Compare:
  - BLIP-2 caption + inversion (baseline)
  - Hand-crafted caption + inversion (human ceiling)
  - PEZ-1 + inversion (proposed)
- **Failure mode.** PEZ doesn't beat captioning. This would mean
  PEZ's reconstruction-aware objective doesn't align with what
  makes a prompt good for DDIM inversion.
- **Risk level.** Low–medium. PEZ is published and known to optimize
  CLIP-image alignment. Whether that translates to inversion fidelity
  is empirical but well-grounded.

### Sub-claim 2 — PEZ-2 (instruction-conditioned) produces correct target prompts

Running PEZ with a three-term joint loss (source-preservation +
instruction-following + warm-start anchor), warm-started from PEZ-1's
solution, produces target embeddings that:

- Preserve PEZ-1's embeddings at positions unaffected by the instruction
- Modify positions affected by the instruction to satisfy its semantic
  intent
- Remain mostly-aligned with PEZ-1 at the per-position embedding level
  (i.e., low cosine distance at most positions)

For a substitution edit `"change the animal into a cat"` against a
source dog photo, PEZ-2 should produce embeddings where one position
(the head-noun position encoding "dog") drifts toward cat-cluster in
CLIP embedding space and the other positions stay near PEZ-1.
For an additive edit `"add a bowtie"`, PEZ-2 should produce
embeddings where one or two positions encode bowtie content while the
rest stay anchored.

- **Measurement.** On a fixed test set of (image, instruction, expected
  edit type) triples, measure:
  - **Per-position embedding stability**: fraction of positions where
    the cosine distance between PEZ-1 and PEZ-2 embeddings stays
    below threshold τ (default 0.05). High stability (>70%) means
    warm-start is doing its job.
  - **Edit semantic correctness** (human raters): does PEZ-2's prompt
    describe an image satisfying the instruction? (Project to nearest
    vocabulary at inspection time for human readability.)
  - **CLIP similarity** of PEZ-2's pooled output to a hand-crafted
    "ground-truth" target prompt for each test case.
- **Failure mode.** PEZ-2 either drifts too far from PEZ-1 (low
  stability, P2P alignment breaks) or doesn't drift enough (low
  edit correctness, instruction not applied). This is the λ vs. γ
  tuning question.
- **Risk level.** Medium. The mechanism is sound but the operating
  point is empirical. R2 characterizes (λ, γ) ranges that work.

### Sub-claim 3 — P2P compose with PEZ-1/PEZ-2 outputs unchanged

Existing P2P and local-blend machinery — designed for natural-
language prompts — composes with PEZ-1 / PEZ-2 outputs without
modification at the K/V level. The only adapter needed is replacing
LCS-over-token-IDs with per-position cosine-distance alignment, since
our outputs are continuous embeddings rather than discrete IDs.
**This is the architectural claim.**

- **Why we believe it.** P2P's K/V swap operates on continuous CLIP
  hidden states regardless of how the prompt was produced. Warm-start
  ensures source and target prompts have the same length and per-
  position correspondence by construction. Per-position cosine
  distance trivially identifies which positions drifted (unmapped,
  driving local-blend masking) vs. which stayed close (matched,
  receiving P2P injection).
- **Measurement.** End-to-end editing quality on substitution and
  additive edit splits (Section 4.4). Compare PEZ-1/PEZ-2 + P2P
  against:
  - Vanilla TI + P2P (broken baseline — TI's N=1, no per-position
    structure, no warm-start anchor)
  - BLIP-2 caption + hand-crafted target + P2P (natural-language
    ceiling, with human in the loop)
  - InstructPix2Pix (instruction-trained baseline)
- **Failure mode.** PEZ outputs work for inversion but P2P edits
  on top produce poor structural preservation. Would suggest PEZ-2's
  per-position drift pattern doesn't isolate cleanly under cosine-
  threshold alignment, or that the cross-attention columns aren't
  spatially localized enough for LocalBlend to mask cleanly.
- **Risk level.** Medium. Compositional argument is sound; empirical
  confirmation needed.

### Risk map

| Sub-claim | Risk | If it fails |
|---|---|---|
| 1 — PEZ-1 > captioning | Low–medium | Fall back to BLIP-2 for source; PEZ-2 still works on top of caption-derived source |
| 2 — PEZ-2 produces correct targets | Medium | Tune (λ, γ) more carefully, or fall back to user-provided target descriptions |
| 3 — P2P compose with PEZ outputs | Medium | Investigate per-position drift patterns; may require an explicit position mask or per-position L_anchor weighting |

The project's contribution lives or dies on **sub-claims 2 and 3**.
Sub-claim 1 is supporting (better source representation).

- Sub-claim 1 fails → caption-source + PEZ-2 still gives instruction-
  conditioned editing. Smaller contribution but still novel.
- Sub-claim 2 fails → no automated instruction following; user has to
  manually describe target prompts. Project pivots to "PEZ as
  source representation for attention editing" — Option 1 from our
  earlier discussion.
- Sub-claim 3 fails → core architecture broken; investigate per-
  position constraints in PEZ-2 or fallback alignment mechanisms.

### What success enables

If all three sub-claims hold, the editing pipeline becomes:

```
Per edit:
  1. PEZ-1 on source image (cached) → source embeddings [N, 768]
  2. PEZ-2 on (source image, user instruction), warm-started from
     PEZ-1 → target embeddings [N, 768]
  3. DDIM-invert source under source embedding's CLIP encoding
  4. Run P2P edit:
     - Per-position cosine distance partitions positions into
       matched (close to source) and unmapped (drifted)
     - Cross-attention swap copies source attention at matched positions
     - Unmapped positions drive local blend mask
  5. Decode → edited image
```

The whole pipeline is built from one PEZ algorithm (run twice with
different conditioning), the existing inversion code, and the
existing P2P machinery. **No new attention controllers, no LLM,
no rule-based parser, no learned task-specific model.** This is the
demonstration in Section 11.

## 3. The core method

The architecture has three core pieces:
- **3.1**: PEZ-1 — source-image inversion to continuous prompt embeddings
- **3.2**: PEZ-2 — instruction-conditioned target generation
- **3.3**: P2P integration

Plus one supporting subsection:
- **3.5**: Why this fixes additive editing structurally

And one summary subsection:
- **3.6**: Computational requirements
- **3.7**: End-to-end pipeline recipe

Section 3.0 below lays out the **three rule-based edit modes** that
gate which mechanism in §3.1–§3.7 the system actually runs. Mode 1
(REPLACE) is the v1 implementation target; modes 2 (ADD) and 3
(EXPLICIT_REPLACE) are future work, mechanically sketched here so the
v1 design doesn't paint itself into a corner.

Each subsection maps to one or more of the sub-claims in Section 2.

### 3.0 Edit modes (rule-based, user-supplied)

The system exposes four modes. The user picks a mode explicitly via
a config flag (`EditConfig.mode`); the system does **not** auto-detect
edit type from the instruction text. This is a deliberate trade-off
against an LLM- or heuristic-classifier: CLIP can't reliably parse
"is this an addition or a substitution?" and the joint loss alone is
brittle on ambiguous cases (see Section 8a). Putting the mode in the
user's hands eliminates a class of "did the system guess my intent?"
failures.

| Mode | User input | Position scope | LocalBlend | Status |
|---|---|---|---|---|
| **REPLACE** | Target descriptor: `"cat"`, `"brown fur"` | All N drift jointly | OFF | **v1** |
| **ADD** | Object noun: `"bowtie"` | First N anchored + K free slots | **ON** | Future (R5) |
| **EXPLICIT_REPLACE** | Source-target pair: `"dog → cat"` | One position unfrozen, N-1 frozen | OFF | Future (R6) |
| **STYLE** | Style descriptor: `"whimsical"`, `"oil painting"` | All N drift jointly (low γ_anchor preset) | OFF | Future (R7) |

Three architectural mechanisms underlie the four modes:

1. **Free-position drift, no LocalBlend** (REPLACE + STYLE — different (λ, γ) presets).
2. **Single-position drift, no LocalBlend** (EXPLICIT_REPLACE).
3. **N anchored + K free slots, with LocalBlend** (ADD).

LocalBlend is **ADD-mode-specific by design**. It exists to spatially
gate *new* cross-attention content that has no source-region prior
(the additive object's column wasn't in PEZ-1's optimization, so its
attention can leak across the image without a mask). REPLACE and
EXPLICIT_REPLACE don't need LocalBlend: the drifted positions
inherited their cross-attention spatial pattern from PEZ-1's
optimization, which already localized them to specific source regions.
STYLE doesn't use LocalBlend either — there's no localized region
because the edit is global. See §3.3 for the LocalBlend mechanism
detail and Appendix A for the API.

PEZ-1 is mode-agnostic — same alternating R=2 algorithm for all four
modes. The mode flag steers PEZ-2's optimization setup and the
editing-time configuration.

#### REPLACE (v1)

- **Input.** A descriptor of desired post-edit content (not an
  imperative — "cat" or "a sleeping cat", not "change the dog to a
  cat"). For best results, phrase as a *complete* description of
  the target image content, not just the changed-content noun (see
  §8a — richer pooled embeddings cancel out shared axes in the
  cosine gradient and concentrate drift on the actually-different
  axes).
- **PEZ-2 setup.** All N positions of `pez_1_embeddings` are unfrozen
  and used as the warm-start. The 3-term joint loss (§3.2) runs as
  specified.
- **`prompt_length`.** **Maximize N** (project default: 15). REPLACE
  mode never needs free slots — it works by drift, not extension —
  so every position is source-detail capacity. The proposal's
  detail-richness argument (§3.2) is fully active in this mode.
- **LocalBlend disabled.** Drifted positions' cross-attention is
  naturally region-localized (inherited from PEZ-1's optimization),
  so spatial gating is unnecessary. Forcing it on would over-mask.
- **Failure modes.** If the source has multiple subjects or the
  target word is far from any PEZ-1 position in CLIP space, the
  joint loss may fail to localize the edit cleanly (every position
  drifts slightly, contaminating non-target content). EXPLICIT_REPLACE
  is the principled fallback for those cases (future).

#### ADD (future, R5)

- **Input.** A noun describing the object to add.
- **PEZ-2 setup.** Variable-length: `prompt_length_pez_2 = N + K`.
  The first N positions are warm-started from `pez_1_embeddings` and
  anchored normally. The last K positions are warm-started from CLIP
  padding-token embeddings and have **no L_anchor pull**. The new
  content lands in the K free slots; the original N stay near PEZ-1.
- **Why this works.** Without free slots (REPLACE-style), an additive
  instruction has nowhere natural to land — the optimizer must
  displace existing source content to make room, sacrificing detail.
  Free slots with zero anchor pull are *cheaper* to modify than
  anchored slots, so the optimizer routes additive content there.
- **LocalBlend enabled.** This is the only mode that uses LocalBlend.
  The new content's cross-attention has no spatial prior from PEZ-1
  (it didn't exist there), so without a mask it could leak across
  the image. LocalBlend's `target_token_indices = the K free-slot
  positions`; the mask gates target-attention to render new content
  in a focused region while preserving source structure elsewhere.

#### EXPLICIT_REPLACE (future, R6)

- **Input.** A source-target pair `(source_word, target_word)`.
- **Localization.** Encode `source_word` through CLIP's text encoder
  → 768-dim vector `e_src`. For each position `i` in
  `pez_1_embeddings`, compute `cos_sim(e_src, pez_1_embeddings[i])`.
  Pick the position(s) with highest similarity (top-1 by default;
  top-k for repeated subjects).
- **PEZ-2 setup.** All positions frozen (`requires_grad=False`)
  *except* the localized one. L_instr targets `target_word`'s CLIP
  encoding. L_source still applies; L_anchor is mostly irrelevant
  since other positions are explicitly frozen.
- **LocalBlend disabled.** Same reasoning as REPLACE — the localized
  position's cross-attention is region-localized from PEZ-1.
- **Why this is more reliable than REPLACE.** When the user knows
  exactly what's in the source and what they want it to become, an
  explicit pair eliminates the "which position should drift?"
  question that REPLACE's joint loss has to answer implicitly. The
  multi-subject failure mode REPLACE has is structurally absent.
- **Trade-off vs. REPLACE.** Loses REPLACE's serendipitous detail-
  discovery (the "PEZ-2 finds bombay cat for a black husky"
  property from §3.2) — explicit replacement honors only the
  literal target word, not the closest source-matched specialization.
  Use REPLACE when you want detail richness; use EXPLICIT_REPLACE
  when you want reliability and explicit control.

#### STYLE (future, R7)

- **Input.** A style descriptor: `"whimsical"`, `"oil painting"`,
  `"low-light photography"`, `"black and white"`. Distinct from
  content/attribute substitution (REPLACE) — style edits affect the
  *whole* rendering, not a localized subject region.
- **PEZ-2 setup.** Same as REPLACE structurally (all N positions
  unfrozen, 3-term joint loss), but with **different
  hyperparameter presets**: `γ_anchor` lower so all positions can
  drift toward style-axis (the desired behavior for global edits),
  `cross_replace_steps` lower so the target prompt drives more of
  the denoising rendering.
- **LocalBlend disabled.** Same reasoning — the edit isn't localized,
  so masking would defeat the purpose.
- **Why this is structurally different from REPLACE.** REPLACE's
  defaults are calibrated to encourage *minimal* drift (preserve
  most positions, edit a few). STYLE's defaults flip that: drift
  *all* positions, that's the point. Conflating the two via a single
  REPLACE mode would either under-edit on style (LocalBlend masks
  too much, anchor too tight) or over-edit on content (everything
  drifts, table-becomes-cat-style contamination). The mode separation
  encodes which behavior the user wants.
- **Why we may eventually want self-attention manipulation here.**
  Style is conventionally handled in attention literature via
  self-attention or AdaIN-style feature swapping. v1 is P2P-only
  (cross-attention manipulation only — see proposal scope). STYLE
  mode in R7 will start with cross-attention only (just lowered
  `cross_replace_steps`). If empirically that's insufficient, R7
  may re-introduce self-attention injection in PnP-style for STYLE
  mode specifically.

### 3.1 PEZ on the source image (sub-claim 1)

**What it does.** Replaces BLIP-2 captioning with continuous-PEZ to
produce a source prompt directly optimized for *recovering the image
through our pipeline*, not for from-scratch generation.

**Algorithm.** Continuous PEZ maintains a soft prompt — N continuous
768-dim embedding vectors — and runs AdamW updates on it directly
(no straight-through projection to vocabulary). The original Wen et
al. 2023 algorithm projects the soft prompt to nearest CLIP
vocabulary tokens at each step; we drop that step. The soft prompt
*is* the output, fed directly to CLIP's text encoder at inference.
We **modify the loss** so that the prompt is optimized for our
actual reconstruction pipeline rather than for matching CLIP-image
similarity (which would be vanilla PEZ).

**The reconstruction-aware loss.** Vanilla PEZ optimizes
`L = -cos_sim(prompt_emb, image_emb)` — "find a prompt that, paired
with random noise, would generate this image under SD." For our use
case, the prompt is paired with DDIM-inverted noise + null-text
optimization, not random noise. So the prompt should encode only what
the inversion+null-text doesn't already capture (semantic content,
identity, style — not geometry, layout, lighting, which are recovered
by `z_T` and the optimized null-text embeddings).

We approximate the gradient of "the prompt + null-text reconstruct
the image under our pipeline" using a **CFG-aware Score Distillation
Sampling (SDS)** surrogate (Poole et al. 2022 for SDS; classifier-free
guidance for the conditioning structure):

```
t      ~ Uniform({1, ..., T})
ε      ~ N(0, I)
x_t    = √α_t · image_latent + √(1 - α_t) · ε

ε_uncond = unet(x_t, t, null_text_embedding)
ε_cond   = unet(x_t, t, prompt_embedding)
ε_cfg    = ε_uncond + cfg_scale · (ε_cond − ε_uncond)

L_sds    = MSE(ε_cfg, ε)
```

Two U-Net forward+backward passes per PEZ step (uncond + cond). The
gradient on the soft prompt comes through `ε_cond`. When `null_text`
already explains an image property (geometry, lighting), `ε_uncond`
already gets that part right — the marginal gradient on the prompt
for that property is small. The prompt's optimization is implicitly
pushed toward content null-text doesn't capture.

This is roughly **600× cheaper per gradient step than running the full
50-step DDIM denoising chain** while still being pipeline-aware (the
`null_text` input makes it reflect our actual inversion+null-text
+ denoising stack, not from-scratch generation).

**Hyperparameters.**
- `N` (prompt length): **prefer large N** (e.g., 12–20). Each
  additional token gives PEZ room to encode another visual detail —
  and the SDS-CFG loss specifically focuses gradient pressure on
  details null-text doesn't already capture, so the token budget is
  spent more efficiently than under vanilla PEZ.
- `num_steps`: 1000–3000.
- `cfg_scale`: 7.5 standard. Higher values give sharper SDS gradients
  but risk DreamFusion-style over-saturation.
- `timestep_sampling`: `uniform` (default) or `importance` (focus on
  mid-timesteps where SDS signal is strongest).

**Alternating optimization of prompt and null-text.** PEZ-1's SDS
loss requires `null_text_embedding`, and null-text optimization
requires a prompt — chicken-and-egg. We resolve it via a two-round
alternating-optimization pipeline (block coordinate descent) that
**naturally enforces the geometry-vs-semantics partition** without
any new hyperparameters tuned per image:

```
# Round 0 — bootstrap from vanilla PEZ
c_0 = vanilla_pez(image)                             # CLIP-only, no SD
                                                      # ~1500 CLIP steps

# Round 1 — null-text fills the gap, prompt refines
z_T, traj = ddim_inversion(image, c_0)
N_0       = null_text_optimization(traj, c_0)        # existing code
                                                      # ~500 SD passes
c_1       = sds_pez(image, N_0, warm_start=c_0)      # NEW: SDS-PEZ
                                                      # with frozen N_0
                                                      # ~1500 SD passes

# Round 2 — re-balance partition
N_1       = null_text_optimization(image, c_1)       # ~500 SD passes
c_2       = sds_pez(image, N_1, warm_start=c_1)      # ~1500 SD passes
                                                      # (or fewer for refinement)

# Final outputs:
prompt    = c_2
null_text = N_1
```

**Residual parameterization for SDS rounds (round-1+ anchoring).** The
SDS rounds (1 and 2 above) need an additional implementation detail
that the schematic above hides: how the warm start `c_(r-1)` is
*preserved* during round `r`'s SDS optimization. Without explicit
preservation, two pathologies appear:

1. **AdamW's default `weight_decay` decays the soft prompt toward the
   origin in CLIP embedding space, not toward the warm start.** This
   is silently anti-anchoring: each step pulls the soft prompt away
   from `c_(r-1)` (which is at some specific non-zero point) toward
   the origin, and only SDS gradient counteracts the pull. The
   converged solution sits at the balance between SDS-optimum and
   origin, biased toward origin by `weight_decay × lr`. In Wen et
   al.'s original PEZ this is fine — there's no warm start to
   preserve — but in our alternating R=2 setup, the default
   actively destroys the round-(r−1) → round-r warm start.
2. **SDS gradient noise causes random-walk drift even when `c_(r-1)`
   is already near-optimal.** SDS estimates have variance from random
   timestep `t` and noise `ε` sampling. Without a restoring force,
   the soft prompt's variance grows linearly with step count
   (`Var(soft_prompt_t) = O(lr² · σ² · t)`), so noisy SDS gradients
   accumulate as drift even if the true SDS gradient at `c_(r-1)`
   is zero.

The fix is a **residual reparameterization**:

```
P    = c_anchor + Δ          # c_anchor frozen (= round-(r-1) output)
Δ_0  = 0                     # Δ is the new optimization variable
optimize Δ via AdamW with weight_decay = δ_wd
```

At Δ = 0 the pipeline reproduces classic-PEZ-on-`c_anchor` exactly,
so we recover the warm start as a literal initialization. AdamW's
weight_decay term, applied to Δ, decays Δ → 0 — which means
P → c_anchor. This converts AdamW's default "pull toward origin"
into "pull toward c_anchor," using the existing weight_decay knob.

**Algebraic equivalence.** Decaying Δ → 0 with AdamW gives:

```
Δ_{t+1} = Δ_t − lr · adam_step(grad) − lr · δ_wd · Δ_t
P_{t+1} = c_anchor + Δ_{t+1}
        = P_t − lr · adam_step(grad) − lr · δ_wd · (P_t − c_anchor)
```

— identical to "decay P toward c_anchor with strength δ_wd," but
implemented via the optimizer rather than a `||P − c_anchor||²` term
in the loss. The optimizer-side route is cleaner because AdamW's
decoupled weight decay doesn't interact with momentum the way a loss
term would (see Loshchilov & Hutter 2019 for the AdamW vs. Adam-with-
L2 distinction).

**Noise-variance bound (the practical payoff).** With weight_decay
λ = δ_wd, Δ becomes a discrete-time AR(1) (Ornstein-Uhlenbeck)
process. At equilibrium under pure noise (no signal):

```
Var(Δ_∞) ≈ lr · σ² / (2λ)         ← saturated, NOT growing with t
```

vs. λ = 0 where `Var(Δ_t) = lr² · σ² · t` grows linearly. Persistent
SDS signal `g` still moves Δ to `g/λ` (bounded but proportional to
the true gradient), so signal direction is preserved while the noise
floor is capped. The practical effect: at the SDS optimum, the prompt
sits *close to c_anchor with bounded noise variance*; far from the
optimum, Δ moves coherently in the signal direction.

**Three properties this gives the alternating algorithm:**

1. **Fixes the anti-anchoring default.** No silent decay toward origin.
2. **Bounded SDS-noise drift.** `c_(r-1)` becomes a stable equilibrium
   of round `r`'s optimization, not just a starting point.
3. **Residual-connection property.** At δ_wd = ∞ the pipeline reduces
   to "do nothing"; at δ_wd = 0 the pipeline reduces to classic-PEZ
   on `c_anchor`. δ_wd interpolates between "trust c_anchor fully"
   and "trust SDS fully" — Δ activates only insofar as SDS genuinely
   demands a correction.

**Round 0 (vanilla PEZ from random init) is unchanged.** No warm
start exists at round 0, so there's nothing to anchor to — the
existing `weight_decay` (decaying toward origin) is the standard
Wen-et-al regularization and stays as-is. The residual parameterization
applies only to rounds 1+ (the SDS-PEZ refinement steps). This is
why the implementation needs two independent config knobs:
`weight_decay` (round-0 vanilla regularization, ≈ 0.1, Wen et al.
default) and `delta_weight_decay` (round-1+ anchor strength on Δ,
empirical sweep — see Open Question 5).

**Why this anchor lives in the optimizer, not in the loss.** PEZ-2
also has an anchor (the `L_anchor = ||soft_prompt − soft_prompt_init||²`
term in §3.2's three-term loss), so a natural question is whether to
unify them. We don't, intentionally:

- PEZ-2's `γ_anchor` is a Knob-1 ablation parameter swept across
  {0.01, 0.1, 1.0} — it's load-bearing for the §4.2 narrative as an
  *explicit* loss coefficient, easier to interpret as a knob.
- PEZ-1's `δ_wd` is a fixed-default thing whose only job is to
  preserve the warm start across rounds; it's not a tuning knob the
  user is expected to sweep.

Same mechanism, two different roles. Keeping them as distinct
implementations preserves the clarity of the proposal's narrative
without compromising correctness — both routes implement the same
mathematical effect at their respective stages.

**PEZ-2's symmetric fix.** The anti-anchoring pathology described
above (AdamW's `weight_decay` decaying toward origin) is *not* unique
to PEZ-1 — PEZ-2 has the same issue any time `weight_decay > 0`,
and it's worse in PEZ-2 because at step 0 (when `soft_prompt = init`)
the L_anchor gradient is zero but `weight_decay` is *immediately*
pulling away from `init` toward origin. A nonzero AdamW `weight_decay`
would silently drift the soft prompt toward origin in the early
steps and contaminate the (λ, γ) ablation grid (§4.2). The fix is
the symmetric one: **PEZ-2 hardcodes AdamW's `weight_decay` to 0.0**
so `L_anchor` is the *sole* regularizer. This is not a config knob —
exposing it would invite users to retune it, reintroducing the
pathology. The decision is captured in `instruction_conditioned.py`
inline.

**Why this enforces the geometry partition (priority 1).** Each step's
natural bias aligns to push geometry into null-text and semantics into
the prompt:

- **Vanilla PEZ is biased toward semantic content.** It optimizes
  CLIP-image cosine similarity. CLIP was trained on caption-image
  pairs; the loss preferentially rewards semantic descriptors
  (subject identity, attributes, style) over geometric specifications
  (precise layout, lighting). The bootstrap prompt `c_0` therefore
  leans semantic.
- **Null-text optimization fills in the residual.** Given a
  semantic-leaning prompt, the gap between predicted and actual noise
  is concentrated on geometric/structural content. Null-text picks
  that up.
- **SDS-PEZ refinement (with frozen null-text)** further pushes the
  prompt toward the semantic content null-text didn't already capture.

After R=2 rounds, both `prompt` and `null_text` are at (approximately)
a fixed point of the alternating optimization — each is optimal given
the other.

**Why this generalizes across inputs (priority 2).** No new
hyperparameters introduced beyond what's published in vanilla PEZ
(Wen et al. 2023) and null-text optimization (Mokady et al. 2022) and
PEZ-1's SDS loss. All three sub-routines have settled defaults that
work across image domains. The only knob R1 ablates is `R` itself
(number of rounds), with `R=2` as the proposed baseline and `R∈{1, 3}`
as cheap-to-run comparisons.

**Cost.** Approximately 2× SDS-PEZ-alone, dominated by two
null-text-optimization calls (~500 SD passes each) plus two SDS-PEZ
calls (~1500 SD passes each). Total per source image: ~25-50 minutes
on an A100. Each step is a known-working component, so failure modes
are isolated and recoverable.

**Inference-time human-readable projection (optional).** For logging
or debugging, snap each soft-prompt position to its nearest CLIP
vocabulary embedding and decode the resulting token IDs as a string.
This is a *one-shot* operation at inference, never during
optimization. The decoded string is illustrative only — the canonical
PEZ-1 output is the continuous `[N, 768]` tensor. Don't make pipeline
decisions based on the projected string.

**Geometry-partition diagnostic.** After PEZ-1 produces `(prompt,
null_text)`, generate from null-text alone (CFG=1, no prompt) starting
from `z_T`:

```
img_from_null = ddim_denoise(z_T, null_text, cfg=1.0)
```

Inspect the result:

- *Partition working*: `img_from_null` shows the source's layout,
  composition, lighting, and rough structure — but with generic /
  wrong identity for the subject. (E.g., for a "black husky"
  source, `img_from_null` shows a dog-shaped silhouette in the right
  pose with the right lighting but a generic-looking dog.)
- *Null-text collapse*: `img_from_null` faithfully reconstructs the
  source including identity. Null-text has captured everything; the
  prompt is doing nothing. R=2 should not produce this; if observed,
  fall back to R=1 or investigate the SDS-PEZ refinement.
- *Partition didn't form*: `img_from_null` looks unrelated to the
  source. Null-text didn't capture image-specific content. Likely a
  bug in null-text optimization or the bootstrap prompt was
  pathological.

Running this diagnostic on every test image during R1 development
gives direct empirical confirmation of priority 1.

**Why we don't use joint optimization with regularizers.** That
alternative was considered and rejected because:

1. **It doesn't structurally enforce the geometry partition.** Two
   regularizers (`λ_N`, `λ_inform`) prevent collapse but don't
   constrain *what kind of content* goes where. Null-text could drift
   to encode discriminative semantic content within its budget; the
   prompt could end up with geometric content. The partition would
   emerge from optimization dynamics that depend on initialization
   and hyperparameter geometry — neither a robust nor a generalizable
   property.
2. **Two new tunable hyperparameters per domain.** No published
   priors. Sweeping `(λ_N, λ_inform)` per image takes hours.

Joint optimization remains an interesting v2 exploration if
alternating proves insufficient — see Section 8 Open Question 1.

**Caching.** PEZ on a source image is expensive (~15–30 min on GPU,
longer for large N). For research we cache outputs per image — once
you've inverted a test image, subsequent edits on it reuse the cached
prompt.

### 3.2 PEZ-2: instruction-conditioned target generation (sub-claim 2)

This subsection describes PEZ-2 as run in **REPLACE mode** (the v1
implementation target — see §3.0). The same machinery underlies ADD
and EXPLICIT_REPLACE; what changes per mode is which positions are
unfrozen during optimization and whether `prompt_length` is extended
beyond N. See §3.0 for the per-mode setup.

**What it does.** Runs PEZ a second time with two simultaneous CLIP
similarity targets — the source image and the user's target descriptor
— producing target embeddings that satisfy both.

**The loss.** Three terms with different roles. The source-preservation
term mirrors PEZ-1's reconstruction-aware loss (CFG-aware SDS with
null-text); the instruction-following term is text-text CLIP cosine;
the warm-start anchor is L2 in soft-prompt space.

**Null-text from PEZ-1, frozen.** PEZ-2 reuses the jointly-optimized
null-text produced by PEZ-1's run on the same source image. Null-text
is **frozen** during PEZ-2 (only the soft prompt is updated). This
keeps the source-preservation term well-defined and ensures the
SDS-CFG loss is comparable to what PEZ-1 saw — same conditioning
regime, same approximation of the editing-time pipeline.

```python
# Inputs computed once before optimization:
instr_emb  = clip_text_encoder(instruction_text)     # [768] for L_instr
null_text  = pre_computed_null_text_embedding        # [1, 77, 768] for L_source

# Optimization variable — the soft prompt:
soft_prompt = pez_1_embeddings.clone()               # [N, 768]  (warm start)
soft_prompt_init = soft_prompt.detach().clone()      # frozen reference

# At each gradient step:
# L_source: SDS-CFG (same form as PEZ-1's loss)
t        ~ Uniform({1, ..., T})
ε        ~ N(0, I)
x_t      = √α_t · image_latent + √(1 - α_t) · ε
ε_uncond = unet(x_t, t, null_text)
ε_cond   = unet(x_t, t, soft_prompt)
ε_cfg    = ε_uncond + cfg_scale · (ε_cond − ε_uncond)
L_source = MSE(ε_cfg, ε)

# L_instr: text-text CLIP cosine (the instruction is text, not image)
prompt_pooled = clip_text_encoder(soft_prompt).pooled
L_instr  = -cos_sim(prompt_pooled, instr_emb)

# L_anchor: L2 in soft-prompt space
L_anchor = ||soft_prompt - soft_prompt_init||²

# Combined:
L = L_source + λ · L_instr + γ · L_anchor

# Backprop through CLIP/U-Net; AdamW step on soft_prompt; repeat.
# No vocabulary projection — soft_prompt is the canonical output.
```

Three loss terms, three roles:

1. **Source preservation** (`L_source`): SDS-CFG surrogate for
   "prompt + null-text reconstruct the source image through our
   pipeline." Same loss family as PEZ-1; same null-text used.
2. **Instruction following** (`L_instr`): pulls the prompt toward
   describing the desired post-edit outcome. The instruction is text;
   text-text CLIP similarity is the appropriate signal.
3. **Warm-start anchor** (`L_anchor`): keeps the soft prompt close
   to PEZ-1's per-position embeddings unless an instruction pressure
   overrides. This is what gives us the "minimal edit" property —
   most positions stay at PEZ-1's embeddings; only positions under
   strong instruction pressure shift in 768-dim space.

**Why this is not the same as classical instruction-following.**
CLIP doesn't really understand instructions like
`"change the animal into a cat"` as actions. It encodes them as
descriptions, and the encoding is dominated by the *target / changed
content* (`cat`) rather than by action verbs (`change`).

The PEZ-2 optimization exploits this: it finds a continuous embedding
sequence that matches both the source image and the (target-content-
dominated) instruction. The result naturally describes the source's
composition with the changed content swapped in. There's no parsing
of the instruction's structure required — the semantic content of the
instruction does the work via CLIP's similarity space.

**What does instruction encoding actually contain.** Worth being
explicit because this is where the empirical claim lives:

| Instruction | CLIP encoding dominantly captures |
|---|---|
| `"change the animal into a cat"` | "cat" (the target subject) |
| `"add a bowtie"` | "bowtie" (the new object) |
| `"turn its fur brown"` | "brown fur" (the new attribute state) |
| `"make it black and white"` | "black and white" (the style descriptor) |
| `"remove the dog"` | weaker — CLIP handles negation poorly |

For the first four (substitution, addition, attribute change, style
change), the joint optimization works because the dominant signal in
the instruction's encoding is exactly what we want to add to/swap into
the source. For removal-style edits, this approach has limitations.

**The level at which optimization happens.**

| Level | What lives there | Role in PEZ-2 |
|---|---|---|
| Soft prompt | `[N, 768]` continuous vectors | The optimization variable; gradients land here; **also the canonical output** |
| Pooled / projected (joint space) | `[768]` | Where the L_instr similarity loss is computed |

The soft prompt is initialized directly from PEZ-1's `[N, 768]`
embeddings (warm start, identity copy). The CLIP transformer
contextualizes the soft prompt, pools at EOS, and projects into joint
space. The L_instr loss compares pooled prompt against pooled
instruction in joint space (where CLIP was trained). No vocabulary
projection: the soft prompt is the optimization variable AND the
final output.

**Hyperparameters that need ablation in R2:**

- **`λ` (instruction strength)**: trades source preservation against
  instruction following. Too low → no edit. Too high → prompt drifts
  off source structure. Likely operating range 0.5–3.0.
- **`γ` (warm-start anchor)**: trades flexibility against alignment
  preservation. Too low → prompt may drift far from PEZ-1's tokens,
  breaking P2P alignment. Too high → no edit happens. Likely range
  0.01–1.0.
- **`num_steps`**: fewer than PEZ-1 because we're refining, not
  searching from scratch. ~300–800 plausible.

**What PEZ-2 outputs in our running examples** (illustrated by
projecting each position's embedding to its nearest CLIP vocabulary
token for human readability — the canonical outputs are continuous
`[N, 768]` tensors):

For source image of a brown dog and instruction `"change the animal
into a cat"`:
```
PEZ-1 (projected):  [a, photo, of, a, fluffy, brown, dog, sitting, grass]
PEZ-2 (projected):  [a, photo, of, a, fluffy, brown, cat, sitting, grass]
                                              ^^^
                            position 6 drifted in 768-dim space;
                            others stayed near PEZ-1's embeddings
```

For instruction `"add a bowtie"`:
```
PEZ-1 (projected):  [a, photo, of, a, fluffy, brown, dog, sitting, grass]
PEZ-2 (projected):  [a, photo, of, a, fluffy, brown, dog, sitting, bowtie]
                                                            ^^^^^^^^^^^^^^^
                            one or two positions drifted to encode
                            bowtie content (which positions depends on
                            loss landscape, not always the trailing ones)
```

For instruction `"turn its fur brown"` (suppose source had a black dog):
```
PEZ-1 (projected):  [a, photo, of, a, fluffy, black, dog, sitting, grass]
PEZ-2 (projected):  [a, photo, of, a, fluffy, brown, dog, sitting, grass]
                                              ^^^^^
                              attribute position drifted; rest stable
```

These projections are illustrative; PEZ-2's actual `[N, 768]` outputs
sit in continuous CLIP embedding space (often between vocabulary
points, encoding visual specifics more precisely than any single
discrete token would). R2 characterizes per-position drift patterns
under different (λ, γ) settings.

**The detail-richness advantage at large N.** A subtle but important
property of this formulation: with a long prompt budget (large N),
PEZ-2 automatically discovers visual properties of the desired edit
that a user would not think to specify in a hand-crafted prompt.

Consider an example (projected-to-vocab strings shown for readability;
PEZ outputs are continuous embeddings that may sit between vocabulary
points and encode visual specifics more precisely than any one token):

```
Source image:    A photo of a black husky with thick fur on a wooden table.

PEZ-1 (projected): "a high resolution photograph of a black husky with
                    thick fur sitting on a wooden table under soft natural
                    lighting"

User instruction:  "change the dog into a cat"

PEZ-2 (projected): "a high resolution photograph of a black bombay cat
                    with long fur sitting on a wooden table under soft
                    natural lighting"
```

The breed `bombay` (a black, fluffy cat) is not something the user
typed — they only said `"a cat"`. PEZ-2's joint optimization finds an
embedding *near* "bombay" in CLIP space (likely even between vocabulary
points, encoding "black-fluffy-cat-axis" more precisely than the
single token "bombay") because:

1. The **source-similarity term** in PEZ-2's loss is still active
   even after warm-start. It pulls the prompt toward describing the
   actual source image — black, fluffy, large — not just a generic cat.
2. The **instruction-similarity term** pulls toward "cat" specifically.
3. The **warm-start anchor** keeps the surrounding context tokens
   stable.

The intersection of these three pressures, in CLIP embedding space,
finds the cat breed whose embedding best satisfies all three
simultaneously — i.e., the cat breed that *visually matches* the
source husky most closely. CLIP knows this implicitly because it was
trained on millions of (image, caption) pairs labeling specific cat
breeds.

A user typing `"a black cat on a wooden table"` would specify color
but not breed — and SD would generate a random black cat. PEZ-2
finds the visually-closest cat breed by pure embedding-space
arithmetic. **The system does the edit better than the user could
have specified manually.**

This argues for using **large N** in PEZ-1 and PEZ-2 (e.g., N=12 or
even higher). Each token slot is detail capacity. Larger N means
more visual properties of the source get encoded into PEZ-1, and more
of those properties carry into PEZ-2's target via the joint
optimization. The token budget is the detail budget.

This is a key part of the "hard edits made easy" framing: not just
that the user gives a simple instruction and the system handles
parsing, but that the system fills in details the user wouldn't have
known to provide.

### 3.3 P2P integration (sub-claim 3)

**Minimal modifications.** P2P's K/V swap mechanism operates on
continuous CLIP hidden states and is agnostic to whether the prompt
was produced by tokenization-from-text or by continuous-PEZ
optimization. The only adapter needed is replacing LCS-over-token-IDs
(the previous default in
[cross_attention.py:64-80](attention_control/cross_attention.py#L64-L80))
with **per-position cosine-distance alignment** between PEZ-1 and
PEZ-2 embeddings. This is strictly simpler than LCS — no insertions,
deletions, or reorderings to handle, since warm-start enforces
identical length and per-position correspondence by construction.

**Why it works.**

- **Same length, same per-position initialization** between PEZ-1 and
  PEZ-2 (warm-start). Position `i` in source corresponds to position
  `i` in target by construction. No alignment search needed.
- **Per-position cosine distance** identifies which positions drifted
  during PEZ-2's optimization:
  ```
  for i in range(N):
      if cos_sim(pez_1_emb[i], pez_2_emb[i]) > τ:
          matched.append(i)         # P2P injects source → target
      else:
          unmapped.append(i)        # P2P leaves alone; new content
                                    # renders via target's organic
                                    # cross-attention
  ```
  τ ≈ 0.95 (cosine similarity) is a reasonable starting point.

The threshold τ is the only new hyperparameter relative to the prior
LCS design. LCS had implicit choices (gap penalties, tie-breaking);
the cosine threshold replaces those with one explicit knob.

**LocalBlend is ADD-mode-specific.** The original P2P paper (Hertz et
al. 2022) introduced LocalBlend specifically for additive edits —
new content being added to the prompt has no source-region prior in
PEZ-1's optimization, so its cross-attention can leak across the
image. LocalBlend builds a spatial mask from the additive content's
cross-attention to gate that leakage.

For REPLACE and EXPLICIT_REPLACE modes, drifted positions inherited
their cross-attention spatial pattern from PEZ-1's optimization,
which already localized them to specific source regions. Adding
LocalBlend on top would over-mask without gain. For STYLE mode the
edit is global; spatial gating defeats the purpose. So **LocalBlend
is enabled only in ADD mode** (see §3.0); the other three modes
disable it. Appendix A specifies the LocalBlend mechanism; the v1
implementation builds it but only ADD mode (R5, future) actually
constructs and attaches an instance.

**Why not classical mixture-of-experts.** A reasonable architectural
alternative is composed classifier-free guidance (Liu et al. 2022):
encode source and target prompts separately, predict noise under each
independently, combine at each diffusion step:

```
ε_combined = α · ε_source + β · ε_target + (1−α−β) · ε_uncond
```

We don't use this because:

1. It generates from scratch — doesn't preserve source structure.
   Would still need DDIM inversion + P2P on top to get editing
   behavior.
2. It blends globally — no spatial localization. Local blend (used
   in ADD mode) provides the spatial mechanism this approach lacks.
3. SD has one generator. "Experts" in our setting are conditioning
   regions in the 77×768 prompt tensor, not separate models. Our
   P2P-based setup IS an MoE — at the cross-attention level, with
   token positions as experts and (in ADD mode) local blend as
   spatial gating — matching SD's architecture natively.

### 3.4 Note on the prior "bounded refinement" phase

A previous version of this proposal included a separate **bounded
continuous refinement** phase that added small perturbations
`Δ_i ∈ ℝ^768`, `||Δ_i|| ≤ ε`, to discrete-PEZ outputs in order to
recover fidelity beyond what discrete vocabulary expressed. This
phase is now subsumed by PEZ-1 and PEZ-2 themselves: with continuous
optimization throughout, there is no discrete vocabulary ceiling to
break out of, and per-position embeddings are already free to sit
anywhere in `ℝ^768` from the start.

The relevant fidelity-vs-stability lever is now PEZ-2's `γ_anchor`
(L_anchor coefficient): high γ keeps embeddings near PEZ-1 (better
P2P alignment, weaker edits); low γ allows larger drift (stronger
edits, alignment risk). This is the same Pareto trade-off the prior
ε knob was meant to characterize, expressed as a single parameter
inside the existing optimization rather than as a separate
post-processing phase.

### 3.5 Why this fixes additive editing

The two failure modes from Section 1's "Additive editing is the
hardest case" subsection are addressed structurally:

**Failure mode 1 (local-blend mask collapse) — fixed because:** PEZ-1
and PEZ-2 are optimized under a reconstruction-aware loss (SDS-CFG
with null-text). The loss landscape rewards per-position embeddings
whose cross-attention contributes constructively to recovering the
source image — which means each position's cross-attention column
ends up spatially localized rather than smeared. Vanilla TI doesn't
have this property because it optimizes only for reconstruction at
N=1; with no per-position structure to spread across, the single S*
necessarily smears its contribution. With N≈15 and reconstruction
gradient flowing through SD's U-Net, each position's cross-attention
column is shaped to a coherent spatial role.

**Failure mode 2 (P2P injection erasure) — fixed because:** warm-
start gives source and target the same length and per-position
correspondence by construction. Edits live at the positions where
PEZ-2's embedding drifted from PEZ-1's — and at those positions,
per-position cosine distance exceeds the threshold τ, marking them
*unmapped*. Unmapped means P2P does NOT inject source content onto
target's column. The new content stays at its position, untouched
by the alignment mechanism.

For substitution edits in **REPLACE mode** (`dog → cat`):
- Most source/target positions have low cosine distance → matched →
  P2P injects.
- The shifted position has high cosine distance → unmapped → P2P
  leaves alone → cat-direction embedding's cross-attention renders
  the cat in the (PEZ-1-localized) animal region. **No LocalBlend
  needed** — the drifted position's cross-attention inherited its
  spatial pattern from PEZ-1's optimization.

For additive edits in **ADD mode** (`+ a bowtie`):
- Most original positions stay at PEZ-1's embeddings (anchor pull) →
  matched.
- The K free-slot positions encode bowtie content → unmapped by
  construction (initialized to padding, drifted to encode the new
  object) → free to render new content.
- **LocalBlend is essential here** — the K free-slot positions had
  no PEZ-1 spatial prior (they didn't exist there), so without a
  mask their cross-attention could leak across the image. The mask
  is built from the K positions' cross-attention and gates rendering
  to a focused region.

In both cases, the architecture's correctness comes from the warm-
start property keeping PEZ-2 close to PEZ-1 except where the
instruction demands change, and per-position cosine distance
identifying that "except where" automatically. ADD mode adds
LocalBlend on top because additive content uniquely needs spatial
gating (no source-region prior to inherit). REPLACE doesn't.

### 3.6 Computational requirements summary

No model training. The base models (CLIP, SD U-Net, VAE) stay frozen
throughout. Three optimization procedures:

| Operation | Cost | Frequency |
|---|---|---|
| PEZ-1 on a source image | ~15–30 min GPU | Per source image (cached) |
| PEZ-2 (instruction-conditioned, warm-started) | ~5–10 min GPU | Per (image, instruction) pair |

Per-edit total: ~20–40 min on first edit of a source image,
~5–15 min on subsequent edits of the same image (PEZ-1 cached).

PEZ-2 is faster than PEZ-1 because warm-starting means we're refining
a near-correct solution rather than searching from scratch.

### 3.7 End-to-end pipeline recipe

Given:
- A real source image
- A user instruction in natural language
  (e.g., `"change the animal into a cat"`, `"add a bowtie"`,
  `"turn its fur brown"`, `"make it black and white"`)

Pipeline:

```
PER-EDIT:

  1. PEZ-1 on source image (cached per source image):
     image → continuous source embeddings src_emb [N, 768]
     (also outputs per-timestep null_text)

  2. PEZ-2 on (source image, instruction), warm-started from PEZ-1:
     - Initialize soft prompt as src_emb.clone()
     - Optimize with three-term loss:
         L = L_source(soft_prompt, image, null_text)
             + λ · L_instr(soft_prompt, instruction)
             + γ · ||soft_prompt - soft_prompt_init||²
     → continuous target embeddings tgt_emb [N, 768]

  3. Run each [N, 768] embedding sequence through CLIP's text encoder
     (pad to 77 with EOS/padding tokens) → 77×768 contextual encoding
     for source and target.

  4. DDIM-invert source image under source contextual encoding
     (use existing src/inversion.py)

  5. Per-position cosine alignment between src_emb and tgt_emb:
     - matched   = [i if cos_sim(src_emb[i], tgt_emb[i]) > τ]
     - unmapped  = [i if cos_sim(src_emb[i], tgt_emb[i]) ≤ τ]
     τ ≈ 0.95 by default.

  6. Run editing denoising loop with [source, target] batch:
     - CrossAttentionController copies source attention columns to
       target at matched positions
     - LocalBlend uses target_token_indices = unmapped positions to
       build a spatial mask; injection is gated by the mask outside
       its boundary

  7. Decode final latent → edited image
```

The crucial property: **all existing attention-editing machinery
works unchanged at the K/V level.** Warm-start gives source and target
the same length and per-position correspondence; per-position cosine
distance produces the matched/unmapped partition that drives P2P and
LocalBlend. The positions that drifted are exactly the edit regions,
and they're naturally identified as unmapped — which is what local
blend needs to localize the edit.

The only adapter relative to the original P2P design: replace
LCS-over-token-IDs with the per-position cosine threshold. No
modifications to the K/V swap mechanism, the denoising loop, or
LocalBlend itself.

### 3.8 Deviations from prior work (and rationale)

This section enumerates the project's deliberate deviations from the
hyperparameters / mechanisms of the source papers (PEZ, P2P,
DreamFusion). Listed for honest scoping and to pre-empt reviewer
"why didn't you follow X's recommendation?" questions.

**From Wen et al. 2023 (PEZ):**
- **`prompt_length`: 4-16 typical → 75 (the CLIP-77 hard cap).** Wen
  et al. used short prompts because their goal was concept inversion
  for a single image. Our goal is full source-scene representation
  with multi-subject capacity, requiring much more per-position
  budget. See §3.0's REPLACE-mode notes.
- **Straight-through vocabulary projection: dropped.** We're
  continuous-PEZ throughout; the soft prompt is the canonical output.
  Architectural decision to avoid the discrete-token-projection's
  Voronoi-jump nonlinearities and to allow free continuous embedding
  drift. See §3.1.
- **Other PEZ machinery (`learning_rate=0.1`, `weight_decay=0.1`,
  AdamW, `batch_size=1`):** ✓ unchanged from Wen et al.
- **`num_steps`: 1000-3000 → 3000 (cap):** at the high end of Wen
  et al.'s range, with adaptive early stopping shortcircuiting when
  convergence is detected. Higher cap accommodates the larger N.

**From Poole et al. 2022 (DreamFusion / SDS):**
- **`cfg_scale`: 100 → 7.5.** DreamFusion's high CFG was for
  *from-scratch text-to-image generation* (sharp gradient direction
  needed). Our SDS is a *reconstruction-aware loss* paired with our
  editing pipeline's CFG=7.5 — the SDS gradient should match the
  editing-time CFG regime. Intentional and project-specific.
- **Timestep sampling: ✓ DreamFusion-style truncation followed.**
  `timestep_sampling: uniform_truncated` drops the t-edge timesteps
  per [0.02·T_train, 0.98·T_train]. Was previously `uniform`
  (vanilla); changed because edge timesteps make the SDS gradient
  unstable. See §3.1's loss specification.

**From Hertz et al. 2022 (Prompt-to-Prompt):**
- **`cross_replace_steps` default: ✓ Hertz's word-swap value (0.4)
  followed.** Earlier versions used 0.8 (Hertz's *additive*-edit
  default), which biased toward conservative edits. Now corrected
  to 0.4 for v1's REPLACE mode; ADD mode (R5) will override to 0.8
  when it lands.
- **`layer_indices` (which cross-attention layers): ✓ all (None
  default).** Matches Hertz's recommendation.
- **Self-attention injection (PnP, Tumanyan et al. 2023): omitted
  in v1.** The proposal scoped this out (see Section 6 reuse map).
  For substitution edits this is a tolerable compromise. **For
  STYLE mode (R7), this is a real loss** — style is conventionally
  handled in attention literature via self-attention or AdaIN-style
  feature swapping. R7 will start cross-attention-only with lowered
  `cross_replace_steps`; if that's insufficient, R7 will re-introduce
  PnP self-attention specifically for STYLE mode. The architecture
  is designed to accommodate this re-introduction without disturbing
  the REPLACE / ADD / EXPLICIT_REPLACE machinery.

**From Mokady et al. 2022 (Null-text Inversion):**
- ✓ Followed: per-timestep null-text optimized for faithful
  reconstruction; used at edit time in the unconditional CFG branch.
- The proposal's PEZ-1 alternates between null-text optimization
  and SDS-PEZ refinement — an extension on top of Mokady, not a
  deviation from it.

## 4. Evaluation plan

### 4.1 Source-image inversion quality (sub-claim 1)

For a held-out set of 20 source images, measure inversion+reconstruction
fidelity under three source-prompt strategies:

- **BLIP-2 caption**: as in the v1 plan. Caption then DDIM-invert.
- **Hand-crafted caption**: a human writes a careful description.
  Establishes a ceiling for caption-based methods.
- **PEZ-1 (continuous embeddings)**: the proposed source representation.

Metrics: PSNR / SSIM / LPIPS on `recon = decode(reconstruct(invert(image,
prompt)))`.

### 4.2 PEZ-2 target prompt quality (sub-claim 2) — Knob 1: divergence

This evaluation measures **Knob 1**: how far PEZ-2's prompt diverges
from PEZ-1, controlled by `(λ_instruction, γ_anchor)`. The sweep
characterizes the operating range of Knob 1 independent of editing-
time settings.

For a fixed test set of 30 (source image, instruction) pairs covering
substitution, addition, and attribute change, measure:

- **Per-position embedding stability**: fraction of positions where
  cosine_similarity(pez_1_emb[i], pez_2_emb[i]) > τ for the
  alignment threshold τ (0.95 default). Higher = warm-start working.
  Target: ≥70% for substitutions, ≥85% for additions (where most
  positions are unchanged).
- **Edit semantic correctness** (human raters, paired comparisons):
  does PEZ-2's output (projected to nearest vocab for inspection)
  describe an image satisfying the instruction?
- **CLIP similarity** between PEZ-2's pooled output and a hand-crafted
  ground-truth target prompt for each test case.
- **(λ_instruction, γ_anchor) ablation**: sweep `λ_instruction ∈ {0.5,
  1.0, 2.0}` and `γ_anchor ∈ {1.0, 0.1, 0.01}` — a 3×3 grid. Drops
  λ=5.0 (the most aggressive end where alignment tends to break,
  not informative beyond confirming the failure mode). Report
  performance across the grid; identify recommended operating points.

Three reference Knob-1 operating points to use across downstream
evaluations:

| Setting | `λ_instruction` | `γ_anchor` | Expected behavior |
|---|---|---|---|
| Conservative | 0.5 | 1.0 | High preservation; instruction barely applied |
| Moderate | 1.0 | 0.1 | Balanced; preservation high, edit visible in prompt |
| Aggressive | 3.0 | 0.01 | Low preservation; large prompt drift toward instruction |

Expected finding: the **moderate** point is the right Knob-1 default;
conservative and aggressive bracket the failure modes (no edit vs.
broken alignment).

### 4.3 P2P composability (sub-claim 3)

For 30 (source image, instruction) pairs from the test set, run:

- PEZ-1 + PEZ-2 + P2P edit (proposed full pipeline)
- PEZ-1 + hand-crafted target prompt (human override of PEZ-2) +
  P2P edit (controls for PEZ-2 quality)

Compare structural preservation (SSIM outside edit region) between
the two. If PEZ-2 produces target prompts that compose with P2P
as well as hand-crafted targets do, sub-claim 3 holds. If structural
preservation degrades with PEZ-2 vs. hand-crafted, the issue is
PEZ-2 producing target prompts whose per-position drift pattern
doesn't align cleanly under cosine-threshold alignment — investigate
threshold τ, drift uniformity across positions, or per-position
L_anchor weighting.

### 4.4 End-to-end editing quality — Knob 2: edit aggressiveness

End-to-end editing quality with the full architecture vs. baselines,
plus a sweep over **Knob 2** — `(cross_replace_steps, self_replace_steps)`
— which controls how much P2P injects source attention vs. lets the
target prompt drive rendering.

**Test set.**

- 50 pairs covering substitution (e.g., dog→cat, young→old),
  addition (+ bowtie, + sunglasses), attribute change
  (turn fur brown), and style change (make it black and white).

**Configurations.**

For each test pair, run the **proposed pipeline** with **moderate
Knob 1** (`λ_instruction=1.0, γ_anchor=0.1`) under three Knob-2
operating points:

| Knob-2 setting | `cross_replace_steps` | `self_replace_steps` | Expected behavior |
|---|---|---|---|
| Subtle | 0.8 | 0.5 | Strong source preservation; edit may not render |
| Moderate | 0.5 | 0.3 | Balanced — recommended default |
| Loud | 0.3 | 0.1 | Weak source preservation; aggressive edit; structural risk |

Plus the standard baseline configurations (run at the moderate
Knob-2 setting only):

- **BLIP-2 caption + hand-crafted target + P2P**: human-in-the-loop
  ceiling for caption-based editing.
- **Vanilla TI + P2P (broken baseline)**: TI's N=1 continuous concept
  injected at the target position; no warm-start anchor, no per-
  position structure.
- **InstructPix2Pix** (Brooks et al. 2023): instruction-trained
  end-to-end editing model.

**Metrics.**

- **Edit quality** (human raters, paired comparisons): "which edit
  better realizes the instruction?"
- **Structure preservation** (SSIM on regions outside the edit mask).
- **Concept fidelity** (CLIP similarity between edited region and
  reference images of the target concept).
- **Local-blend mask quality** (additive edits only): IoU between
  generated mask and hand-annotated ground-truth edit region.

**Hypothesis.**

- Proposed method at moderate Knob-2 beats vanilla TI + P2P
  decisively on additive edits.
- Proposed method matches or beats InstructPix2Pix on structure
  preservation. Concept fidelity should be comparable.
- Proposed method approaches the BLIP-2 + hand-crafted ceiling on edit
  quality.
- Knob 2 has predictable trade-offs: subtle → high structure
  preservation but low concept fidelity; loud → high concept fidelity
  but degraded structure preservation. Moderate sits at the elbow.

### 4.5 Full (λ, γ, cross_replace_steps) sweep

Knob 1 is two-dimensional in its own right (`λ_instruction` and
`γ_anchor` are independent — see §3.2's discussion of why they're
not redundant). Combined with Knob 2's `cross_replace_steps`, the
joint operating space is **3D: λ × γ × cross_replace_steps**.

The project's contribution is on the Knob-1 side — the 2D (λ, γ)
plane is where we expect to characterize PEZ-2's behavior. Knob 2 is
inherited P2P machinery; we sweep it for completeness but the
visualization treats it as a secondary axis (replicated panels rather
than equal-status axes).

**Why we run the full grid despite Knob 2 not being a contribution
axis.** Knob 2's per-cell cost is ~2 min editing on top of an already-
computed PEZ-2 output, so adding |Knob 2| panels is cheap relative to
the |Knob 1| PEZ-2 budget. Generating all 36 cells lets us
demonstrate that the (λ, γ) story holds across reasonable Knob-2
choices (rather than being an artifact of one cross_replace_steps
setting), without committing to Knob 2 as a contribution axis.

**Sweep design.**

For each of 5 representative (image, instruction) pairs in the test
set, evaluate the **3 × 3 × 3 = 27 setting grid**:

- `λ_instruction ∈ {0.5, 1.0, 2.0}` (Knob 1.a, primary axis)
- `γ_anchor ∈ {1.0, 0.1, 0.01}` (Knob 1.b, primary axis)
- `cross_replace_steps ∈ {0.8, 0.5, 0.3}` (Knob 2, secondary axis)

(λ=5.0 from the §4.2 sweep is dropped here — it sits in the
alignment-broken failure region and isn't informative for visualizing
the operating envelope.)

**Visualization is (λ, γ)-centric.** Heatmaps and image grids use λ
and γ as their two axes; cross_replace_steps appears as a third
dimension via:
- **Image grid:** three (λ × γ) panels stacked or shown side-by-side,
  one per cross_replace_steps value. Plus a GIF cycling
  cross_replace_steps as the animation frame, with the **BLIP-baseline
  edit included as a sidecar cell** in each frame so the reader sees
  method-vs-baseline at every cross_replace_steps step.
- **Heatmaps:** the per-position embedding-stability and quality-
  metric heatmaps lay out (λ, γ) as the two visible axes; if a
  heatmap depends on cross_replace_steps it gets the same
  "three-panels-or-GIF" treatment.
- **Per-(λ, γ) strip comparison:** for each operating point a
  cross_replace_steps strip (1×3 row of method edits) is paired
  with the BLIP-baseline strip at the same cross_replace_steps
  values, producing a 13-row mega-figure (12 method strips + 1
  baseline strip × 3 cross_replace_steps cols). This is the
  reader-facing "is this (λ, γ) operating point better than the
  baseline" scan view.

**Cost per (image, instruction) pair:**
- 9 PEZ-2 runs × ~5 min = ~45 min (with `num_steps: 300` cap and
  `pez_search`'s movement-based adaptive early stopping; some
  settings converge well before the cap)
- 27 edits × ~2 min = ~54 min
- Total: ~1.5-2 h per pair

For 5 representative pairs: ~8-10 h of GPU time on a single A100.
Cost-aware ordering (outer loop over (λ, γ), inner over
cross_replace_steps) keeps PEZ-2 amortized — each PEZ-2 output is
reused across all 3 Knob-2 values.

For 5 representative pairs: ~4 hours of GPU time. Scales linearly
with the size of the representative set.

**What to extract from the grid:**

1. **Diagonal of coherence**: which (Knob 1, Knob 2) combinations
   produce visually coherent edits — where the edit renders cleanly
   AND the source structure is preserved.
2. **Failure-mode quadrants**:
   - High Knob 1 + low Knob 2 (aggressive prompt + subtle edit):
     prompt encodes major change but P2P injection prevents
     rendering. Result: subtle visible change despite major prompt
     drift. Diagnostic that the editing mechanism is the bottleneck.
   - Low Knob 1 + high Knob 2 (conservative prompt + loud edit):
     prompt barely differs but editing is permissive. Result:
     subtle change. Diagnostic that the prompt is the bottleneck.
   - Both high: dramatic edit, high structural-breakdown risk.
   - Both low: minimal edit, source preserved.
3. **Generalizable defaults**: if the moderate × moderate cell is
   the best across all 5 representative pairs, declare those as the
   universal v1 defaults. If the optimal cell varies by edit type
   (substitution vs. additive vs. style), document that and pick a
   reasonable compromise.

**Output for the writeup**: a 5 × 9 grid (5 image-instruction pairs ×
9 settings) of edit results. Reading the grid empirically tells the
reader where the proposed pipeline lives in (Knob 1, Knob 2) space
and what the sensitivity profile looks like.

This characterization is the project's contribution to the
*operability* of attention-based editing: instead of "P2P just works"
or "P2P doesn't," we provide a 2D map of the editing space with
characterized regions.

### 4.6 Detail-richness advantage at large N

This validates the claim from Section 3.2's detail-richness subsection:
PEZ-2 with large N produces target prompts containing visual details
the user didn't specify, sourced from the source image's CLIP
similarity. Test:

For 10 source images with hand-craftable simple instructions
(e.g., `"change the dog into a cat"`, `"change the car into a
truck"`), compare:

- **Baseline 1**: User-typed simple target prompt (e.g., `"a photo
  of a cat on a table"`). Generic; no source-specific details.
- **Baseline 2**: User-typed detailed target prompt (e.g., `"a black
  fluffy cat on a wooden table"`). User adds details by reading the
  source image.
- **Proposed**: PEZ-2 output (no human in the loop).

For each, generate the edit and measure:
- **Visual coherence with source** (CLIP image-image similarity
  between source and edit, on regions outside the changed object).
  Tests how well the surrounding context is preserved.
- **Visual coherence within the edit** (CLIP image-text similarity
  between edited object's region and a hand-written description of
  the target with source-matched details).
- **Human rater preference** (paired comparisons).

Hypothesis: PEZ-2 matches Baseline 2 (user adding details) on detail
quality, beats Baseline 1 (sparse user prompt) decisively, and may
even beat Baseline 2 in cases where the user misses subtle source
properties (specific breed identifiers, fine textures).

This is the "**hard edits made easy**" empirical claim: the system
doesn't just match what a user could do — it can do better than
naive user prompts because it has automatic access to all of the
source image's visual properties via the CLIP source-similarity term.

### 4.7 Per-position alignment robustness on PEZ-derived prompts

Verify that the per-position cosine threshold (default τ ≈ 0.95)
produces sensible matched/unmapped partitions on PEZ-2 outputs:

- Source is PEZ-1-discovered (continuous embeddings, possibly far
  from any vocabulary point)
- Target is PEZ-2-discovered, warm-started from PEZ-1 (mostly aligned
  but some positions drifted)
- Multi-attribute edits where multiple positions drift between source
  and target

Failure mode: many positions drift slightly under PEZ-2's optimization
even at unaffected positions, producing a noisy unmapped set. If
observed, increase γ_anchor to keep unaffected positions tightly
anchored, or sweep τ to find the elbow that cleanly separates "drifted
for the edit" from "drifted incidentally."

## 5. Codebase organization

New code lives under `src/pez/`, `src/splice/`. Existing code
(`src/inversion.py`, `attention_control/`, `src/utils.py`) is reused
unchanged at the K/V level; only the alignment helper is replaced
(LCS-over-IDs → per-position cosine threshold).

Proposed layout:

```
src/
  pez/
    __init__.py
    search.py             # core PEZ algorithm: soft prompt + AdamW
                          # (no vocabulary projection; soft prompt = output)
                          # used by both source_inversion and instruction_conditioned
    source_inversion.py   # PEZ-1: image → continuous source embeddings [N, 768]
                          # (caches outputs per image hash)
    instruction_conditioned.py
                          # PEZ-2: (image, instruction) → continuous target embeddings
                          # [N, 768]; warm-started from PEZ-1, three-term joint loss

  splice/
    __init__.py
    align.py              # per-position cosine-distance alignment
                          # between source and target embeddings;
                          # returns (matched_indices, unmapped_target_indices)

  metrics/
    __init__.py
    fidelity.py           # PSNR / SSIM / LPIPS of inversion+reconstruction
    edit_quality.py       # structure preservation, concept fidelity (CLIP),
                          # mask quality (IoU vs ground-truth edit region)

attention_control/
  local_blend.py          # built per Appendix A (needed by R4)

notebooks/
  R1_pez_source.ipynb        # PEZ-1 vs. BLIP-2 captioning fidelity comparison
  R2_pez_instruction.ipynb   # PEZ-2 ablation of (λ, γ); per-position embedding
                             # stability; edit semantic correctness
  R4_full_evaluation.ipynb   # end-to-end editing comparison vs.
                             # IP2P, vanilla TI, caption baselines

data/
  test_images/               # held-out source images for editing
    person_at_desk.png
    dog_on_grass.png
    ...
  test_instructions.json     # paired instructions for each test image
  ground_truth_target_prompts.json
                             # human-written target prompts for evaluation

cache/
  pez_source_prompts/        # cached PEZ-1 outputs ([N, 768] tensors) per
                             # source image (each PEZ-1 run ~15-30min)
  pez_target_prompts/        # cached PEZ-2 outputs per (image, instruction)
                             # (each PEZ-2 run ~5-10min)
```

The two library directories (`pez/`, `splice/`) form a clean
separation: PEZ produces continuous embedding sequences (used both
for source inversion and instruction-conditioned target generation);
splice aligns them for P2P. Each is independently testable.

## 6. Existing-code reuse map

| Component | Status | When needed |
|---|---|---|
| Existing inversion (DDIM + null-text) in `src/inversion.py` | Required, unchanged | All phases that involve image reconstruction. |
| Existing attention controllers (P2P) in `attention_control/` | Required, K/V-swap mechanism unchanged; alignment helper replaced (LCS-over-IDs → per-position cosine threshold) | Phase R4. K/V swap operates on continuous CLIP hidden states regardless of how the prompt was produced. |
| Existing SD utility code in `src/utils.py` | Required, unchanged | All phases. |
| New: `attention_control/local_blend.py` | Required (built when starting R4) | Phase R4. Specification in Appendix A. |
| New: `src/splice/align.py` | Required for R4 | Per-position cosine-distance alignment between PEZ-1 and PEZ-2 embeddings. |
| BLIP-2 captioning wrapper | Used as R1 baseline only | A minimal BLIP-2 wrapper for the source-fidelity comparison in R1. Not part of the proposed pipeline; exists only for the comparison. |

## 7. Phased experimental milestones

The project breaks into three phases, each producing a discrete
artifact. (A previous version had a fourth "bounded refinement" phase;
that is now subsumed by PEZ-1's continuous optimization — see §3.4.)

### Phase R1 — PEZ on the source image (sub-claim 1)

**Goal:** a working continuous-PEZ implementation that takes a source
image and produces an `[N, 768]` continuous embedding tensor, plus
measurements showing PEZ-1 embeddings give higher inversion fidelity
than BLIP-2 captions.

**Files to create:**

- `src/pez/search.py`
  ```python
  def pez_search(
      loss_fn,                                  # callable(soft_prompt) → scalar
      prompt_length: int,
      num_steps: int,
      lr: float,
      seed: int,
      device: torch.device,
      initial_soft_prompt: torch.Tensor | None = None,
  ) -> torch.Tensor:
      """Run continuous PEZ optimization. Returns the final soft prompt
      [prompt_length, 768] in CLIP embedding space.

      Algorithm:
        - Initialize a [prompt_length, 768] soft prompt
          (random, or warm-started from initial_soft_prompt).
        - For num_steps:
          - Compute loss = loss_fn(soft_prompt).
          - Backprop directly to soft_prompt (no vocabulary projection).
          - AdamW step on soft_prompt.
        - Return final soft_prompt as the canonical output.
      """
  ```
- `src/pez/source_inversion.py`
  ```python
  def pez_invert_source(
      image: Image.Image,
      clip_model,
      tokenizer,
      prompt_length: int = 15,
      num_steps: int = 1500,
      cache_dir: Path | None = None,
  ) -> tuple[torch.Tensor, list[torch.Tensor]]:
      """PEZ-1 on a single source image. Returns:
        - source_embeddings: Tensor[N, 768], continuous CLIP-input
          embeddings that reconstruct the source under our pipeline
        - null_text_per_timestep: per-timestep optimized null-text
          (list of T tensors, each [1, 77, 768])
      Caches to disk by image hash.
      """
  ```
- `src/metrics/fidelity.py`
  ```python
  def measure_inversion_fidelity(
      image: Image.Image,
      prompt: str,
      pipeline,
  ) -> dict[str, float]:
      """Run DDIM inversion + null-text + reconstruction.
      Return PSNR, SSIM, LPIPS vs. original image."""
  ```

**What to produce:**

- A working continuous-PEZ implementation, validated on a few test
  images.
- 10 source images with cached PEZ-1 embeddings in
  `cache/pez_source_prompts/`.
- A comparison table per source image:

  | source-prompt strategy | PSNR | SSIM | LPIPS |
  |---|---|---|---|
  | BLIP-2 caption | ... | ... | ... |
  | Hand-crafted caption (ceiling) | ... | ... | ... |
  | PEZ-1 (continuous embeddings) | ... | ... | ... |

**Expected finding:**

PEZ-1 embeddings give comparable or better inversion fidelity than
BLIP-2 captions, particularly on images with hard-to-describe content
(specific patterns, unusual compositions, fine textures). Hand-crafted
captions remain the ceiling but require human effort.

If PEZ-1 doesn't beat BLIP-2 (sub-claim 1 fails), pivot: fall back to
BLIP-2 captioning for source representation. PEZ-2 (R2 onward) still
works on top of caption-derived sources — it just becomes "PEZ-2 from
a caption" instead of "PEZ-2 from PEZ-1". The architecture survives;
the source-representation contribution is reduced.

Estimated time: 1-2 weeks. Mostly adapting Wen et al.'s PEZ codebase
to integrate with the existing inversion pipeline (without the
straight-through vocabulary projection).

---

### Phase R2 — PEZ-2: instruction-conditioned target generation (sub-claims 2, 3)

**Goal:** implement PEZ-2 (instruction-conditioned PEZ with warm-start
from PEZ-1); ablate (λ, γ) hyperparameters; verify P2P composability
with end-to-end edits on a few hand-picked test cases.

**Files to create:**

- `src/pez/instruction_conditioned.py`
  ```python
  def pez_instruction_conditioned(
      source_image: Image.Image,
      instruction: str,
      pez1_embeddings: torch.Tensor,    # [N, 768] from PEZ-1
      null_text: torch.Tensor,          # frozen, from PEZ-1
      clip_model,
      sd_pipeline,
      lambda_instr: float = 1.0,        # instruction strength
      gamma: float = 0.1,               # warm-start anchor
      num_steps: int = 500,
  ) -> torch.Tensor:
      """PEZ-2: optimize a soft prompt warm-started from pez1_embeddings
      with three-term loss (SDS-CFG source preservation + text-text
      CLIP cosine to instruction + L2 anchor to warm-start).

      Loss:
        L = L_source(soft_prompt, image, null_text)         # SDS-CFG
            + lambda_instr * (-cos_sim(pooled_prompt, instr_emb))
            + gamma * ||soft_prompt - soft_prompt_init||^2

      Returns: continuous target embeddings [N, 768]."""
  ```
- `src/splice/align.py`
  ```python
  def align_pez_prompts(
      source_embeddings: torch.Tensor,  # [N, 768]
      target_embeddings: torch.Tensor,  # [N, 768]
      threshold: float = 0.95,          # cosine similarity cutoff
  ) -> tuple[list[int], list[int]]:
      """Per-position cosine-distance alignment between source and
      target embeddings. Warm-start makes both sequences the same
      length with per-position correspondence by construction.

      Returns:
        - matched_indices: positions where cos_sim >= threshold
          (P2P will inject source K/V here)
        - unmapped_target_indices: positions where cos_sim < threshold
          (P2P leaves alone; these drive LocalBlend)"""
  ```

**What to produce:**

- A working PEZ-2 implementation, validated on 5 hand-picked
  (source_image, instruction) test cases with known expected target
  prompts.
- A (λ, γ) ablation grid: for each test case, sweep λ ∈ {0.5, 1.0,
  2.0, 5.0} and γ ∈ {0.01, 0.1, 1.0}. Report:
  - Per-position embedding stability (fraction of positions where
    cos_sim(pez_1_emb[i], pez_2_emb[i]) > τ for τ=0.95)
  - Edit semantic correctness (human/CLIP score against expected target)
- Recommended (λ, γ) operating range for each edit type
  (substitution / addition / attribute change / style change).
- Hand-picked end-to-end editing demos showing PEZ-2's output composed
  with P2P (LocalBlend not yet built, so use no-mask P2P).

**Expected finding:**

A useful (λ, γ) operating region exists where token preservation is
high (>70%) and edit correctness is high. CLIP's instruction encoding
is dominated by target content (cat, bowtie, brown), and the
optimization successfully pulls the warm-started prompt toward
incorporating it.

If no operating region works, sub-claim 2 fails. Investigate:
- Whether removing γ (warm-start anchor) helps
- Whether smaller perturbations to PEZ-1 work via different
  optimization recipes (e.g., projected gradient descent only on
  selected positions)

Estimated time: 2 weeks.

---

### Phase R4 — End-to-end editing evaluation (paper-grade)

**Goal:** end-to-end editing experiments with the full PEZ-1 + PEZ-2
+ P2P architecture vs. baselines; paper-grade writeup.

**Files to create:**

- `src/metrics/edit_quality.py` — implementations of metrics in
  Section 4.4: structural SSIM, concept CLIP similarity, identity
  LPIPS, mask IoU.
- `attention_control/local_blend.py` — built per Appendix A.

**Demo and evaluation notebook: `notebooks/R4_full_evaluation.ipynb`**

For each of 50 (source image, instruction) pairs:

1. Load cached PEZ-1 source embeddings (from R1) and run PEZ-2 (from
   R2) to get target embeddings.
2. Encode source and target prompts through CLIP's text encoder
   directly from their `[N, 768]` continuous embeddings.
3. DDIM-invert source under source encoding.
4. Compute per-position cosine alignment between source and target
   embeddings.
5. Set up controllers:
   - `CrossAttentionController` with the matched/unmapped partition
   - `LocalBlend` with `target_token_indices` = unmapped positions
6. Run editing denoising loop.
7. Save side-by-side comparisons:
   - Source image
   - Edit using **vanilla TI + P2P** (broken baseline)
   - Edit using **InstructPix2Pix** (instruction-trained baseline)
   - Edit using **BLIP-2 + hand-crafted target + P2P** (human ceiling)
   - Edit using **proposed (PEZ-1 + PEZ-2 + P2P)**

**What to produce:**

- 50-pair test set results table with per-metric averages and
  per-edit-type splits (substitution / addition / attribute / style).
- Ablation table:
  - Without PEZ-1 (use BLIP-2 source instead)
  - Without PEZ-2 (use hand-crafted target instead) — measures how
    much PEZ-2 contributes vs. just having a good source representation
  - Cosine threshold τ ablation
- Qualitative figure for the paper.
- Paper draft: motivation, method, results, related work, limitations.

**Expected finding:**

Proposed architecture beats vanilla-TI baseline decisively on
additive edits, matches/beats InstructPix2Pix on structure
preservation, and approaches the BLIP-2 + hand-crafted ceiling on edit
quality (showing PEZ-2 successfully automates the human's role).

Estimated time: 3–4 weeks including writeup iteration.

---

**Total estimated timeline: 5-7 weeks of focused work.**

A reasonable milestone cadence: R1 done by week 1, R2 by week 3,
R4 by weeks 4-7.

---

### Future phases — modes ADD, EXPLICIT_REPLACE, STYLE

The v1 milestones above implement only the REPLACE mode (see §3.0).
The other three modes are future work, mechanically scoped here so
the v1 architecture doesn't preclude them.

#### Phase R5 — ADD mode (variable-length PEZ-2)

**Goal.** Extend PEZ-2 with K extra free slots warm-started from
padding-token embeddings (no L_anchor pull on those slots), so
additive instructions like `"add a bowtie"` land in the free slots
instead of displacing source content.

**Files to modify:**
- `Pez2Config` gets `prompt_length_extra: int = 0` (K). REPLACE mode
  uses K=0; ADD mode uses K ∈ {2, 3, 4} typically.
- `pez_invert_with_instruction` constructs a soft prompt of length
  N+K, warm-starts the first N from `pez_1_embeddings`, the last K
  from CLIP padding-token embeddings. L_anchor only acts on the
  first N rows.
- `EditConfig.mode` accepts `"add"`; `run_p2p_edit` passes ADD's
  K value through and treats the last K positions as guaranteed
  unmapped (LocalBlend mask is built from their cross-attention).

**Estimated time:** 1-2 weeks on top of R4.

#### Phase R6 — EXPLICIT_REPLACE mode (cosine-localized constrained PEZ-2)

**Goal.** When the user supplies `(source_word, target_word)`, localize
the source word to a specific PEZ-1 position via cosine and run
constrained PEZ-2 with all other positions frozen.

**Files to modify:**
- `pez_invert_with_instruction` accepts an optional `source_word: str`
  and `target_word: str` pair. Runs CLIP on the source word, computes
  per-position cosine to PEZ-1, picks top-1 (or top-k via threshold).
- `pez_search` extends to support per-position freezing: a
  `frozen_positions: set[int]` argument that masks gradient on those
  rows during optimization.
- `EditConfig.mode` accepts `"explicit_replace"`; `run_p2p_edit`
  passes the source/target pair through.

**Estimated time:** 1 week on top of R4.

#### Phase R7 — STYLE mode (global rendering shift)

**Goal.** Support style-descriptor inputs like `"whimsical"`,
`"oil painting"`, `"low-light photography"` that shift the *whole*
image's rendering rather than swapping a localized subject. Distinct
from REPLACE because the desired drift pattern is "all positions
move toward style-axis" rather than "few positions move."

**Files to modify:**
- `EditConfig.mode` accepts `"style"`. Validation guard releases.
- `run_p2p_edit` skips LocalBlend (same as REPLACE) and uses
  STYLE-tuned `cross_replace_steps` defaults (lower than REPLACE
  so the target prompt drives more of the rendering).
- `Pez2Config` gains optional STYLE-mode hyperparameter presets
  (or callers override via `_apply_overrides_nested`):
  γ_anchor lower (e.g., 0.01) so all positions can drift; λ
  higher (e.g., 2.0) so the style-axis pull dominates.
- No changes to PEZ-1 or PEZ-2's algorithm — same machinery, just
  different operating point.

**Open question — likely answered by re-introducing PnP.** Cross-
attention manipulation alone is widely understood to be insufficient
for stylistic edits — style is conventionally handled via self-
attention or AdaIN-style feature swapping. The original P2P paper
itself noted this limitation, and Plug-and-Play Diffusion (Tumanyan
et al. 2023) was developed specifically to address it via self-
attention injection. v1 is P2P-only (cross-attention only) by
project scope, but **R7 should plan to re-introduce PnP self-
attention injection from the start** rather than try cross-attention-
only first. The empirical evidence from prior work strongly suggests
cross-attention alone is the wrong tool for global aesthetic shifts;
attempting it first would just re-derive the known limitation.

The re-introduction is contained to STYLE mode — REPLACE / ADD /
EXPLICIT_REPLACE all stay cross-attention-only and don't change.
The architecture is designed for this: the CrossAttentionController
already has `is_cross` switching, and adding a parallel
SelfAttentionController for STYLE is a natural extension.

**Estimated time:** 2-4 weeks on top of R4 (PnP self-attention
implementation + integration with the editing loop's mode dispatch).

These phases (R5, R6, R7) are non-load-bearing for the project's main
contribution (the two-PEZ + P2P architecture), but they materially
expand the supported edit envelope and are natural extensions once
v1 is validated.

## 8. Open research questions

These should be addressed during the project; they are not blockers but
they shape the methodology:

1. **R=2 sufficiency and v2 alternatives to alternating.** The
   v1 baseline uses alternating optimization with R=2 rounds (vanilla
   PEZ → null-text → SDS-PEZ → null-text → SDS-PEZ). Open questions:

   - **Is R=2 enough?** R1 should ablate R∈{1, 2, 3} on a small set
     of test images. R=2 should produce a near-fixed-point
     decomposition; R=3 tests whether further iteration helps. If
     R=1 is sufficient (vanilla PEZ + null-text + frozen-null-text
     SDS-PEZ), we save compute. If R=3 is needed, R1 catches it.
   - **When does the geometry partition fail?** The diagnostic
     described in Section 3.1 should pass for ≥90% of test images.
     If it fails for specific image classes (e.g., highly-stylized
     images, abstract content), document and investigate.
   - **Joint optimization with regularizers (v2).** A cheaper
     alternative is joint optimization of prompt and null-text with
     `λ_N` and `λ_inform` regularizers. Trades reliability for
     compute. Considered for v2 if R=2 alternating is unworkable due
     to compute constraints or insufficient quality.
   - **Curriculum CFG (v2).** Anneal cfg_scale from 1 (no null-text
     needed) to the editing-time value (7.5) during PEZ-1 to give
     the optimization a more stable trajectory. Schedule design is
     non-trivial; deferred unless v1 alternating fails.

2. **Edit-type robustness within the supported instruction taxonomy.**
   Within the instruction types PEZ-2 *can* handle (see Section 8a's
   limitation taxonomy), some edits may still fail or behave
   inconsistently. R1 should empirically characterize: across the
   supported types (substitution, addition, attribute change, style
   change), which specific edits produce sensible target prompts and
   which don't, and what (λ, γ) settings work across the most types.
   This is the empirical "operating envelope" of PEZ-2 within its
   supported scope.

3. **(λ, γ) operating range.** R2 ablates the joint loss
   coefficients. Open question: is there a single (λ, γ) that works
   across edit types, or do different edit types need different
   settings? If the latter, we'd need a way to detect edit type from
   the instruction — without an LLM, this might require user-supplied
   edit-type tags.

4. **Multi-step instructions.** "Change the animal to a cat AND add a
   bowtie" — encode as a single instruction (CLIP encoding handles
   both signals), or sequence as two PEZ-2 runs? Probably the latter
   for cleaner alignment, but worth comparing.

5. **Cosine alignment threshold τ.** Default τ = 0.95 is heuristic.
   Worth sweeping τ ∈ {0.90, 0.95, 0.97, 0.99} on the test set to
   find the elbow that cleanly separates "drifted for the edit" from
   "drifted incidentally."

6. **SDS stability.** DreamFusion-style SDS has known issues
   (over-saturation, mode collapse, sensitivity to timestep sampling).
   For continuous-PEZ — which optimizes a soft prompt in CLIP-input
   embedding space rather than continuous pixels — these issues may
   manifest differently. Worth measuring and possibly mitigating with
   VSD (Variational Score Distillation, Wang et al. 2023) if vanilla
   SDS proves unstable.

7. **Off-distribution risk for CLIP/SD on continuous prompts.** CLIP's
   text encoder was trained on inputs that are at vocabulary points;
   continuous PEZ prompts can drift to arbitrary points in CLIP-input
   embedding space. TI demonstrates this works in practice (the
   manifold is locally smooth), but PEZ-2's instruction-conditioned
   joint loss may push the prompt to less-tested regions. Mitigation:
   monitor per-position L2 distance from the nearest vocabulary
   embedding as a sanity diagnostic; raise alarm if average distance
   exceeds typical TI operating ranges. L_anchor partly hedges
   against this by keeping the prompt near PEZ-1's converged
   embeddings.

8. **Interaction with P+ / per-layer textual inversion.** P+ (Voynov
   et al. 2023) shows per-layer embeddings capture concepts more
   richly. Could PEZ-1 / PEZ-2 extend to the per-layer setting? Likely
   yes but adaptation needed.

9. **Default for `delta_weight_decay` (PEZ-1 round-1+ residual
   anchor).** §3.1's residual parameterization sets the round-1+
   anchor strength via AdamW's weight_decay on Δ. The default is
   currently a guess (0.1, mirroring PEZ-2's γ_anchor magnitude).
   Worth A/B'ing across `δ_wd ∈ {0.0, 0.05, 0.1, 0.5}` on a small
   test set, comparing reconstruction PSNR after each round AND
   `‖c_r − c_(r-1)‖²` (drift magnitude per round). Expected pattern:
   `δ_wd = 0` shows large drift even at SDS optimum (random walk);
   high `δ_wd` shows tight anchoring at the cost of suppressing
   genuine SDS corrections. Pick the elbow that bounds noise drift
   without crippling refinement.

## 8a. Known limitations (not pursued in v1 or v2)

The following are explicit limitations of the proposed architecture
that the project does not aim to resolve. Listed for honest scoping
and so users / reviewers know what to expect.

### Instruction encoding is image-content-biased

PEZ-2's `L_instruction` term uses CLIP text-text cosine similarity
between the prompt's pooled CLIP encoding and the instruction's
pooled CLIP encoding. CLIP was trained on caption-image pairs from
the web — captions describe what's *in* images, not preferences,
intentions, mental states, counterfactuals, comparatives, or modal
statements. The pooled CLIP encoding therefore compresses natural-
language instructions toward "what would an image of this look
like." Modal/intentional words get washed out by content nouns.

**Categories of instruction the system handles well**, mapped to the
four rule-based modes from §3.0:

- **Substitution** (REPLACE mode in v1; EXPLICIT_REPLACE for
  multi-subject or far-from-head-noun cases) — user supplies a target
  descriptor like `"cat"`, or in EXPLICIT_REPLACE a pair like
  `"dog → cat"`.
- **Attribute change** (REPLACE mode) — user supplies an attribute
  descriptor: `"brown fur"`, `"smiling"`, `"old"`. Mechanically
  identical to substitution: drift one or more positions toward the
  target attribute.
- **Addition** (ADD mode, future R5) — user supplies a noun:
  `"bowtie"`, `"sunglasses"`. Variable-length PEZ-2 puts the new
  content in unanchored free slots; LocalBlend gates rendering.
- **Style change** (STYLE mode, future R7) — user supplies a style
  descriptor: `"whimsical"`, `"oil painting"`, `"black and white"`.
  Distinct from REPLACE because the edit is global; LocalBlend is
  disabled and (λ, γ) presets encourage uniform drift across all
  positions.

**Categories that fail or degrade gracefully:**
- **Behavioral / preference / mental-state instructions** — e.g.,
  `"make the animal like eating fish more"`. CLIP's pooled encoding
  is dominated by "fish" (the concrete content noun). The user's
  intent — the animal's preference — is not in CLIP's representation
  vocabulary. The system's behavior is *graceful degradation toward
  the closest image-content interpretation*: it produces a target
  prompt resembling "an animal eating fish," ignoring the modal
  ("like") and comparative ("more") qualifiers entirely. Result is
  a coherent edit semantically adjacent to but not identical to the
  user's intent.
- **Negation** — `"make it not red"`. CLIP doesn't represent negation
  well; the encoding still has "red" as a dominant signal. Likely to
  produce something red rather than something non-red.
- **Counterfactuals** — `"as if X had never happened"`. Modal/
  counterfactual semantics aren't in CLIP. Unpredictable behavior.
- **Comparatives without referents** — `"make it bigger"`. CLIP
  doesn't have a strong "bigger" axis without a comparison anchor.
  Likely no-op or wrong direction.
- **Removal** — `"remove the dog"`. Slightly better than negation
  (CLIP can encode absence weakly via context), but still
  unreliable.

### Why we don't fix this

The fix requires either:

- **An LLM in the pipeline** (translate intent into image-content
  description before PEZ-2). The project explicitly rejects this
  to avoid LLM dependencies.
- **A custom instruction encoder** trained on (instruction, edited
  image) pairs that maps semantic intent directly into editing-
  appropriate embedding space. Major training effort plus needs
  paired data we don't have.
- **Fine-tuning CLIP on edit-instruction language**. Same data
  problem; significant compute cost.

None of these is in the project's scope for v1 or v2. The graceful-
degradation behavior is the cost of avoiding an LLM dependency.

### How users should phrase inputs

The mode-based design (§3.0) constrains user inputs to specific
shapes per mode, which sidesteps most phrasing issues:

- **REPLACE mode** takes a target descriptor — a content/attribute
  noun phrase, not an imperative, and *not a style*. `"cat"` ✓,
  `"a sleeping cat"` ✓, `"change the dog to a cat"` ✗ (the verb
  confuses CLIP), `"whimsical"` ✗ (this is STYLE, not REPLACE).
- **ADD mode** (future R5) takes an object noun — `"bowtie"` ✓,
  `"add a bowtie"` ✗.
- **EXPLICIT_REPLACE** (future R6) takes a pair — `("dog", "cat")`,
  not a sentence.
- **STYLE mode** (future R7) takes a style/aesthetic descriptor —
  `"whimsical"` ✓, `"oil painting"` ✓, `"low-light"` ✓,
  `"black and white"` ✓, but NOT a content noun (route those to
  REPLACE).

This is what makes "rule-based modes" a meaningful design choice
over the prior "auto-detect from instruction text" approach: the
mode flag tells the system *how* to interpret the input, and the
input shape per mode is constrained enough that CLIP encoding
behaves predictably.

For requests outside the supported modes (modal, comparative,
counterfactual, behavioral) — see the failure-categories list above
— users should rephrase as a description of desired content:

| Instead of | Phrase as (and pick mode) |
|---|---|
| `"make the animal like eating fish more"` | ADD mode: `"a fish in front of the animal"` |
| `"make it not red"` | REPLACE mode: `"blue"` (or whatever specific color) |
| `"make it bigger"` | REPLACE mode: `"a giant version of X"` if size relative to a referent is the intent |
| `"remove the dog"` | REPLACE mode with a content target: `"an empty backyard"` (describe the post-edit content directly) |

Inputs that fit one of the supported modes get strong behavior.
Inputs forced into a mode they don't fit get graceful degradation
to the closest mode-conforming interpretation. This is the expected
operating envelope.

### Edit-type envelope: P2P-style edits only

The framework is designed around **P2P-style edits** — those that can
be expressed as a localized change to a token-anchored prompt
description of the source image. This covers the categories listed
above (substitution, addition, attribute change, style change), but
*not* edits that fall outside P2P's mechanism:

- **Geometric edits** — `"rotate the head 30°"`, `"move the cup to the
  left"`, `"flip the image"`. P2P's cross-attention column swap
  preserves spatial layout from the inversion trajectory; it has no
  mechanism to move or rotate content. These edits are out of scope.
- **Multi-object compositional edits** — `"put the dog on top of the
  cat and the cat on top of the table"`. P2P operates on a single
  prompt-level edit at a time and inherits CLIP's poor handling of
  binding/relations. Out of scope.
- **Edits requiring novel scene structure** — `"add a second dog"`
  (when the source has one), `"a crowd version of this"`. P2P
  preserves the source's structural skeleton via the column swap;
  introducing additional structural elements is at best unreliable
  and typically out of scope.
- **Pixel-precise edits** — `"recolor exactly this region"` with an
  externally-supplied mask. P2P's LocalBlend mask is derived from
  cross-attention to edit tokens, not from a user mask; this
  framework does not expose a pixel-level mask interface.
- **Identity-preserving edits across large semantic changes** —
  `"the same person but as a child"`. Identity preservation through
  large CLIP-distance moves is a separate research thread (DreamBooth,
  Textual Inversion, identity-locking LoRAs); PEZ-1's continuous
  embedding inversion captures identity well at *small* CLIP-distance
  moves but doesn't carry per-instance identity strongly enough for
  large semantic shifts.

The `LocalBlend` mask gives gentle help on **additive** P2P edits
(masks the target's new content from being overwritten by source
attention), but it doesn't expand the envelope to non-P2P edit types.

If your use case requires edits outside this envelope, this framework
is the wrong tool: pick InstructPix2Pix-style end-to-end editing
models (for compositional/geometric edits), DreamBooth or LoRA
fine-tuning (for identity-preserving large edits), or mask-conditioned
inpainting (for pixel-precise region edits).

## 9. Related work and how this differs

| Method | What it does | Why it doesn't solve our problem |
|---|---|---|
| **PEZ / Hard Prompts Made Easy** (Wen 2023) | Soft-prompt optimization with straight-through projection to CLIP vocabulary | We adopt the optimization machinery (soft prompt + AdamW + gradient through CLIP/SD) but drop the vocabulary projection — soft prompt is the canonical output. Wen et al. didn't apply PEZ as a source-representation primitive for attention-based editing, nor extend it to instruction-conditioned target generation, nor pair it with a reconstruction-aware SDS-CFG loss. |
| **Textual Inversion** (Gal 2022) | Learns one continuous-token concept embedding via image-text reconstruction | TI optimizes one 768-dim vector for a concept — no per-position structure, no instruction conditioning, no alignment story for downstream editing. Our design extends to N≈15 vectors with: (a) reconstruction-aware SDS-CFG loss tied to the editing pipeline, (b) instruction-conditioned target generation pass with warm-start anchor, (c) per-position cosine-distance alignment for P2P composition. The single-vector design is what made TI not compose with P2P; our N-vector + warm-start design directly addresses that. |
| **DreamBooth** (Ruiz 2022) | Fine-tunes the U-Net for a specific concept | Modifies model weights; concept not transportable via prompt manipulation |
| **InstructPix2Pix** (Brooks 2023) | Trains a diffusion model end-to-end for instruction-conditioned image editing | Replaces P2P entirely; requires bootstrapped training data via GPT-3; produces a black-box editing model rather than a compositional pipeline |
| **Composable Diffusion** (Liu 2022) | Combines multiple text conditionings via composed CFG | Generates from scratch — no source structure preservation. Wouldn't compose with our P2P setup without adding the same machinery anyway |
| **Custom Diffusion** (Kumari 2022) | Optimizes K/V projections per concept | Composition lives in projection space; doesn't compose with P2P's prompt-side column-swap mechanism |
| **P+** (Voynov 2023) | Per-layer textual inversion | Orthogonal — could combine with continuous PEZ for richer per-layer representations (Open Question 8) |
| **Concept Sliders** (Gandikota 2023) | Trains directional axes between concepts | Different abstraction (continuous attribute axes); uses LoRAs not prompts |
| **Mix-of-Show** (Gu 2023) | LoRA fusion across multiple concepts | LoRA-level composition, not prompt-level |
| **Prompt-to-Prompt** (Hertz 2022) | Cross-attention column swap for editing | Built for natural-language prompts. This proposal extends its applicability to PEZ-derived continuous prompts via per-position warm-start correspondence (alignment falls out of warm-start, not out of token-ID equality). |
| **BLIP-2 captioning** (Li 2023) | Image-to-text via caption | Used as a baseline; PEZ-1 replaces it for source representation, with continuous-PEZ optimizing reconstruction directly under the editing pipeline |

The contribution is **the two-PEZ architecture for instruction-
conditioned editing** — PEZ-1 for source representation, PEZ-2 for
instruction-conditioned target generation, both composing with existing
P2P machinery via per-position warm-start correspondence. The
instruction-following step is embedding-space optimization (PEZ-2's
joint loss) rather than parsing or LLM-based reasoning. No cited
method does this — each solves a related but distinct piece.

## 10. What success looks like

The headline demonstration is **fully-automated instruction-conditioned
real-image editing built from two PEZ optimizations**, with no
captioning, no LLM, no rule-based parser, and no learned task-specific
model — and producing edits with detail richness beyond what naive
user prompts would capture.

A demo where:

1. The user provides a real photo of a black husky with thick fur
   sitting on a wooden table.
2. The user provides a natural-language instruction:
   `"change the animal into a cat"`.
3. The pipeline:
   - **PEZ-1** runs on the source image with N=15 → continuous
     source embeddings `[15, 768]` that capture the source's visual
     details. Projected to nearest vocabulary for human inspection
     this might read like `["a", "high", "resolution", "photograph",
     "of", "a", "black", "husky", "with", "thick", "fur", "on", "a",
     "wooden", "table"]` — but the canonical artifact is the
     continuous tensor.
   - **PEZ-2** runs on `(source image, instruction)`, warm-started
     from PEZ-1's embeddings, with the three-term joint loss:
     ```
     L = L_source(soft_prompt, image, null_text)        # SDS-CFG
         + λ · (-cos_sim(pooled_prompt, pooled_instr))  # text-text
         + γ · ||soft_prompt - soft_prompt_init||²
     ```
     The instruction's CLIP encoding is dominated by `"cat"`. The
     joint optimization finds embeddings that satisfy:
     - Most positions stay near PEZ-1's embeddings (warm-start anchor)
     - One or two positions drift in 768-dim space toward cat-cluster
       (instruction term)
     - The chosen direction in cat-cluster must visually match the
       source — black, thick fur, large breed (source-similarity term)

     Result projected to nearest vocab: `["a", "high", "resolution",
     "photograph", "of", "a", "black", "bombay", "with", "long",
     "fur", "on", "a", "wooden", "table"]`. The breed `bombay` (a
     black, fluffy cat) was *not* in the user's instruction. PEZ-2
     found an embedding near it (likely between vocabulary points,
     even more visually-precise than `bombay` exactly) by satisfying
     source visual properties simultaneously with instruction
     semantics.
   - DDIM-inverts the photo under PEZ-1's source encoding.
   - Runs P2P edit. Per-position cosine alignment marks most positions
     as matched (cosine ≥ τ); the drifted positions (`husky → bombay`,
     `thick → long`) are unmapped → drive the local-blend mask. P2P
     injects at matched positions, preserving the source's structural
     skeleton.
4. Output: **the same composition, in the same pose, on the same
   wooden table, with a black fluffy bombay cat in place of the
   husky** — visually coherent because the cat breed was chosen by
   matching the husky's visual signature, not by random selection.

The research contribution has two parts:

1. **The two-PEZ architecture works end-to-end without any language-
   model component for instruction following.** PEZ-1 replaces
   captioning; PEZ-2 replaces both LLM-based prompt planning and
   rule-based parsing. The whole instruction-interpretation step
   happens in CLIP embedding space via PEZ-2's joint loss
   optimization.

2. **The system fills in details the user wouldn't have specified.**
   With large N, PEZ encodes source visual properties richly, and
   PEZ-2's joint optimization preserves those properties while
   applying the instruction. This is "hard edits made easy" in a
   strong sense: not just automating what the user would type, but
   doing it better than naive user prompts because the system has
   automatic access to source visual properties via embedding-space
   reasoning.

The same architecture handles additive edits (`"add a bowtie"` →
PEZ-2 finds bowtie tokens consistent with the source's lighting and
formality), attribute changes (`"turn its fur brown"` → PEZ-2 swaps
the color attribute), and style changes (`"make it black and white"`
→ PEZ-2 incorporates style descriptors). No edit-type-specific code
paths — the same joint loss optimization handles them all.

## 11. Where to start

Pick up [Phase R1](#phase-r1--pez-on-the-source-image-sub-claim-1) in
section 7. The repo doesn't currently have a continuous-PEZ
implementation; adapting Wen et al.'s public codebase into `src/pez/`
(against the existing SD2.1 + frozen-component setup in
`src/utils.py`) — and *removing* their straight-through vocabulary
projection step — is the entry point. Once PEZ-1 is producing
continuous embeddings that work with the existing DDIM inversion
pipeline, the rest of the project follows naturally.

The first concrete commit should:

1. Create `src/pez/` with module stubs.
2. Adapt PEZ from Wen et al.'s repo into `src/pez/search.py`. Strip
   the vocabulary-projection step. Verify the soft prompt converges
   under SDS-CFG loss and the resulting `[N, 768]` tensor produces
   sensible reconstruction when fed to CLIP's text encoder.
3. Wire up `src/pez/source_inversion.py` with disk caching, since each
   PEZ run is ~15-30 min and we want to iterate fast on downstream
   code.
4. Run PEZ-1 on one test image; pass the resulting embeddings through
   CLIP's text encoder + existing `src/inversion.py` to verify the
   integration. Compare reconstruction PSNR against a BLIP-2 caption
   baseline.

That's the smallest viable starting point. From here, the comparison
table for sub-claim 1 can be filled in, and Phase R2 (PEZ-2
instruction-conditioned generation) becomes a natural next step using
the same PEZ infrastructure with the additional L_instr + L_anchor
loss terms.

---

## Appendix A — LocalBlend specification

Required by Phase R4. Provides spatial gating so attention injection
can be disabled inside an edit region (giving the new content room
to render) and enabled outside (preserving source structure).

### Mechanism

At each denoising step, build a binary mask over the image's spatial
grid by aggregating cross-attention to the target prompt's "edit"
positions (the positions that drifted between source and target — i.e.,
the unmapped target indices from per-position cosine-distance
alignment).

- Inside the mask: skip P2P injection at *all* token positions for
  patches in the masked region. The target's organic cross-attention
  drives those patches.
- Outside the mask: standard P2P injection. The source's attention
  pattern at mapped positions is copied to the target.

### API

```python
class LocalBlend:
    """Mask state for P2P cross-attention injection gating.

    Attached to CrossAttentionController via its local_blend
    parameter. The controller feeds and consults this object to
    decide where to inject.
    """

    def __init__(
        self,
        target_token_indices: list[int],   # unmapped target positions
        threshold: float = 0.3,             # of max activation
        base_resolution: int = 16,          # spatial side length
        dilate_iters: int = 1,              # mask boundary softening
    ):
        ...

    def record_cross_attention(
        self,
        attn_4d: torch.Tensor,   # [batch, heads, spatial, tokens]
    ) -> None:
        """Accumulate the target half's attention to selected tokens
        for the current step. Called by CrossAttentionController."""

    def step(self) -> None:
        """Finalize the current step's mask and store it for use at
        the NEXT step. Reset accumulator. Idempotent (forward-compat
        reserve): if a future v2 re-introduces a second controller,
        both can call this safely."""

    def get_mask(self, spatial_resolution: int) -> torch.Tensor:
        """Return the current mask resampled to the requested
        resolution. Returns None at step 0 (no mask yet)."""

    def reset(self) -> None:
        """Clear all state. Call between editing runs."""
```

### Implementation notes

- **Mask lifecycle**: cross-attention is recorded during the layer's
  forward pass and the accumulator is finalized at end-of-step, so the
  mask used at step t comes from cross-attention recorded at step t-1.
  Step 0 has no mask; injection is unmasked. (Even in the P2P-only
  setup, the lag is structural — the mask is finalized in
  `controller.step()` which is called after the U-Net forward.)
- **Mask resolution**: cross-attention maps come at multiple spatial
  resolutions (8×8, 16×16, 32×32, 64×64 for SD2.1). Compute the mask
  at one canonical resolution (16×16 recommended), aggregate across
  heads and selected tokens, threshold, then resample on-the-fly to
  whatever resolution a consuming layer needs (`F.interpolate`).
- **Threshold normalization**: `mask = mask / (mask.max() + 1e-8)`,
  then `binary = (mask > threshold).float()`. Optional dilation via
  `F.max_pool2d` to soften boundaries.
- **Aggregation**: at each cross-attention layer, slice the target
  half of `attn_4d`, mean over heads, select target_token_indices,
  mean over selected tokens. Reshape per-layer to `[H, W]`. Resample
  to canonical resolution. Average over all visited layers in the step.

### Modifying existing controllers

`CrossAttentionController.__call__`: after computing attention but
before P2P injection, call `local_blend.record_cross_attention(...)`.
Then in `_word_swap`, gate the column copy by the mask:

```python
mask_flat = local_blend.get_mask(spatial)   # [spatial] or None
if mask_flat is None:
    target[:, :, :, tgt_tok] = source[:, :, :, src_tok]
else:
    w = mask_flat.view(1, 1, spatial)        # broadcast
    target[:, :, :, tgt_tok] = (1 - w) * source[:, :, :, src_tok] \
                              + w * target[:, :, :, tgt_tok]
```

`CrossAttentionController.step()` calls `local_blend.step()` to advance
the mask state at the end of each denoising step.

The idempotency guard on `LocalBlend.step()` is forward-compatible
reserve — if v2 re-adds a second controller (e.g., PnP-as-toggle), the
same `LocalBlend` instance can be shared without double-finalization.

---

## Appendix B — Alignment alternatives

The default alignment is **per-position cosine similarity** between
PEZ-1 and PEZ-2 embeddings, gated by a single threshold τ (≈ 0.95).
This works because warm-start gives source and target prompts
identical length and per-position correspondence by construction.

This appendix records two alternatives in case per-position cosine
proves insufficient empirically:

### Alternative 1 — Bipartite matching (for re-ordered drift)

If R4 shows that PEZ-2's optimization sometimes shuffles per-position
roles (e.g., the breed-encoding axis migrates from position 7 to
position 9), per-position cosine fails. Replace with maximum-weight
bipartite matching over the cosine-similarity matrix:

1. Compute `C[i, j] = cos_sim(pez_1_emb[i], pez_2_emb[j])`.
2. Solve Hungarian assignment (`scipy.optimize.linear_sum_assignment`)
   for the best `i → j` mapping.
3. Filter assignments below threshold τ — those are unmapped.

This handles permutations and near-synonyms but breaks the "position
correspondence by construction" invariant. Only use if measurement
shows it's needed.

### Alternative 2 — Pre-CLIP embedding distance vs. post-CLIP contextual distance

Per-position cosine on the *input* embeddings (`pez_*_emb`, [N, 768])
catches drift at the optimization variable. Per-position cosine on
the *contextual* output embeddings (post-CLIP-transformer, [N, 768])
catches drift in how the embeddings are *contextualized*. They can
diverge: the input embedding might drift slightly while the
contextual encoding drifts a lot (or vice versa) because of self-
attention interactions.

Default is to align on input embeddings (cleaner, faster). If R4
shows this misses meaningful contextual drift, switch to contextual
or use both with separate thresholds.

### When to use either alternative

The default per-position-on-input-embeddings should suffice for the
v1 pipeline because (a) warm-start keeps optimization in a small
neighborhood, (b) L_anchor explicitly penalizes per-position drift
in input space, and (c) PEZ-2's typical convergence touches only 1-2
positions for typical edits. The alternatives are stress-test
mitigations, not v1 requirements.
