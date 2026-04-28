# Hard Edits Made Easy: Instruction-Conditioned Discrete Prompt Inversion for Real-Image Editing

> A direct extension of Wen et al. 2023's "Hard Prompts Made Easy"
> (PEZ): we apply discrete prompt inversion to **both** the source-
> representation problem and the instruction-following problem,
> producing a fully-automated real-image editing pipeline from two
> PEZ optimizations:
>
> 1. **PEZ-1** runs on the source image → produces a discrete CLIP-
>    vocabulary prompt that reconstructs the source. Replaces BLIP-2
>    captioning as the source representation.
>
> 2. **PEZ-2** runs on `(source image, user instruction)` jointly →
>    produces a discrete prompt satisfying both the source's visual
>    content and the instruction's semantic intent. Warm-started from
>    PEZ-1's solution so the two prompts share token positions for
>    unchanged content. Instruction following emerges from CLIP's
>    existing semantic alignments via embedding-space optimization —
>    no LLM, no rule-based parser, no learned task-specific model.
>
> A non-obvious advantage of this formulation: with large N (long
> prompts), PEZ recovers visual details the user wouldn't think to
> specify. For `"change the husky into a cat"`, PEZ-2 finds a cat
> breed whose visual signature matches the source husky's coloring
> and fur texture — automatically — by satisfying the source-image-
> similarity term while moving toward the instruction. **The system
> doesn't just automate what a user would do; it does it better than
> the user could.**
>
> Existing P2P attention-editing machinery composes with PEZ-1 /
> PEZ-2 outputs unchanged: they share most token positions (warm-start
> property), so P2P alignment is trivial; positions affected by the
> instruction differ between source and target, becoming the unmapped
> positions that drive local-blend masking.
>
> Optional **bounded continuous refinement** (`||Δ|| ≤ ε`) on PEZ-1
> and/or PEZ-2 outputs gives a tunable fidelity lever without losing
> position-stability.

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
identifiable lexical role in the prompt — it's "this image's gestalt"
collapsed into one 768-dim point. P2P's mechanism (copying attention
columns at matched token positions) needs the source and target prompts
to share interpretable structure: `"dog"` in source aligns to `"dog"` in
target; their cross-attention columns can be swapped because both are
the column for the noun "dog" at a specific position.

If your source prompt is `"a photograph of a S*"` and your target is
`"a photograph of a S* with a T*"`, the only positions occupying real
content are S* and T* — and their cross-attention behavior is whatever
happened to emerge during textual inversion training. Nothing about
the training procedure encourages those columns to be:

- **Localized** — concentrated at one position rather than smeared across
  the contextual encoding
- **Stable** — behaving consistently when the surrounding prompt context
  changes
- **Swappable** — having a structure that mirrors how natural-token
  cross-attention columns behave (so P2P's column-swap operation
  produces sensible results)

This is the gap. Standard textual inversion optimizes only for
*reconstruction fidelity*. It doesn't optimize for any of the properties
that make a learned concept play well with downstream attention editing.

### Our approach: two PEZ optimizations, source and instruction-conditioned

The personalization literature pursues fidelity by moving from
discrete CLIP vocabulary into continuous embedding space. That move is
what causes the smearing in the first place. **We propose the opposite
direction: stay in discrete-token space throughout, and let
instruction-following emerge from CLIP's existing semantic alignments
via embedding-space optimization.**

The pipeline runs **PEZ** (Wen et al. 2023, "Hard Prompts Made Easy")
twice with different conditioning:

1. **PEZ-1: source-image inversion.** Replace BLIP-2 captioning with
   PEZ-on-source. PEZ optimizes a sequence of *real CLIP vocabulary
   tokens* (actual English words) that, when used as a prompt, make
   the model reconstruct the source image. Because the output is real
   CLIP tokens, position-stability is automatic: P2P treats them like
   any other natural-vocabulary tokens.

2. **PEZ-2: instruction-conditioned target generation.** PEZ runs a
   second time with two simultaneous CLIP similarity targets:
   - the source image (preserve visual content)
   - the user's natural-language instruction text (apply edit)
   
   Warm-started from PEZ-1's discovered tokens, the optimization
   produces a target prompt that mostly matches PEZ-1 but with
   instruction-relevant positions modified. This is the key technical
   piece — instruction following emerges from CLIP's pretrained
   semantic alignments through PEZ-2's joint loss, with **no LLM,
   no rule-based parser, no learned task-specific model.**

The two prompts (PEZ-1 → source, PEZ-2 → target) compose with existing
P2P unchanged because:

- Most positions match (warm-start) → P2P alignment is trivial via
  token-ID matching
- Positions that differ are the edits → unmapped, drive local-blend
  masking
- All tokens are real CLIP vocabulary → cross-attention behaves as
  CLIP was trained for

**Optional continuous refinement** (`||Δ|| ≤ ε`) can be applied to
either PEZ output to recover fidelity beyond what discrete vocabulary
allows, with bounded perturbations that preserve position-stability.

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
   spatial room. The mask is built from
   cross-attention to the new concept's column. If that column is
   smeared (the bowtie's "signal" leaks across the encoding), the
   cross-attention pattern is spatially diffuse. The mask either
   over-covers (loses structure) or under-covers (no room for the
   concept). Either way, the additive edit fails.

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

**This makes additive editing the empirical showcase for the
proposed architecture.** If discrete-prompt + corpus-expert composition
materially improves additive edit quality vs. continuous-TI injection,
the contribution is real and measurable. If it only helps replacement
edits, the contribution is marginal because replacement was already
mostly working.

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

PEZ run on the source image yields a discrete prompt that, when used
for DDIM inversion, reconstructs the source image at higher fidelity
than a BLIP-2 caption of the same image.

- **Measurement.** PSNR / SSIM / LPIPS on source images reconstructed
  via DDIM inversion + null-text optimization. Compare:
  - BLIP-2 caption + inversion (baseline)
  - Hand-crafted caption + inversion (human ceiling)
  - PEZ-1 + inversion (proposed)
  - PEZ-1 + bounded continuous refinement (`||Δ|| ≤ ε`)
- **Failure mode.** PEZ doesn't beat captioning. This would mean
  PEZ's CLIP-image-similarity objective doesn't align with what
  makes a prompt good for DDIM inversion.
- **Risk level.** Low–medium. PEZ is published and known to optimize
  CLIP-image alignment. Whether that translates to inversion fidelity
  is empirical but well-grounded.

### Sub-claim 2 — PEZ-2 (instruction-conditioned) produces correct target prompts

Running PEZ with a joint loss `L = -cos_sim(prompt, image) - λ ·
cos_sim(prompt, instruction)`, warm-started from PEZ-1's solution,
produces target prompts that:

- Preserve PEZ-1's tokens at positions unaffected by the instruction
- Modify positions affected by the instruction to satisfy its semantic
  intent
- Remain mostly-aligned with PEZ-1 at the token-ID level

For a substitution edit `"change the animal into a cat"` against a
source dog photo, PEZ-2 should produce something like
`[a, photo, of, a, fluffy, brown, cat, sitting, grass]` with `dog`
swapped for `cat`. For an additive edit `"add a bowtie"`, PEZ-2
should produce `[..., wearing, a, bowtie]` with the source tokens
preserved and bowtie tokens appended (or inserted at a sensible
position).

- **Measurement.** On a fixed test set of (image, instruction, expected
  edit type) triples, measure:
  - **Token preservation rate**: fraction of PEZ-1's tokens that
    appear at the same position in PEZ-2's output. High preservation
    (>70%) means warm-start is doing its job.
  - **Edit semantic correctness** (human raters): does PEZ-2's prompt
    describe an image satisfying the instruction?
  - **CLIP similarity** of PEZ-2's output to a hand-crafted "ground-
    truth" target prompt for each test case.
- **Failure mode.** PEZ-2 either drifts too far from PEZ-1 (low
  preservation, P2P alignment breaks) or doesn't drift enough (low
  edit correctness, instruction not applied). This is the λ vs. γ
  tuning question.
- **Risk level.** Medium. The mechanism is sound but the operating
  point is empirical. R2 characterizes (λ, γ) ranges that work.

### Sub-claim 3 — P2P compose with PEZ-1/PEZ-2 outputs unchanged

Existing P2P and local-blend machinery — designed for natural-
language prompts — composes with PEZ-1 / PEZ-2 outputs without
modification. **This is the architectural claim.**

- **Why we believe it.** PEZ outputs are real CLIP vocabulary tokens.
  Source and target prompts share most positions (warm-start). P2P
  alignment via token-ID matching trivially maps shared positions;
  unchanged positions get attention swap; modified positions are
  unmapped and drive local-blend masking.
- **Measurement.** End-to-end editing quality on substitution and
  additive edit splits (Section 4.4). Compare PEZ-1/PEZ-2 + P2P
  against:
  - Continuous TI + P2P (broken baseline)
  - BLIP-2 caption + hand-crafted target + P2P (natural-language
    ceiling, with human in the loop)
  - InstructPix2Pix (instruction-trained baseline)
- **Failure mode.** PEZ outputs work for inversion but P2P edits
  on top produce poor structural preservation. Would suggest PEZ-2's
  outputs have structural properties that fight against P2P's
  alignment expectations.
- **Risk level.** Medium. Compositional argument is sound; empirical
  confirmation needed.

### Sub-claim 4 — Bounded continuous refinement gives a useful Pareto frontier

Adding small perturbations `Δ_i ∈ ℝ^768` with `||Δ_i|| ≤ ε` to PEZ-1
(and optionally PEZ-2) outputs improves reconstruction fidelity
without breaking position-stability or P2P composability.

- **Measurement.** Sweep ε for both PEZ-1 and PEZ-2 outputs:
  - Reconstruction fidelity (PSNR / SSIM / LPIPS)
  - Footprint concentration in the contextual encoding
  - End-to-end editing quality
- **Failure mode.** No useful frontier — either fidelity barely
  improves before stability collapses or vice versa.
- **Risk level.** Low. This is a smaller, optional fidelity boost
  on top of the core (PEZ-1 + PEZ-2) architecture. If sub-claim 4
  fails, drop refinement; the main pipeline still works.

### Risk map

| Sub-claim | Risk | If it fails |
|---|---|---|
| 1 — PEZ-1 > captioning | Low–medium | Fall back to BLIP-2 for source; PEZ-2 still works on top of caption-derived source |
| 2 — PEZ-2 produces correct targets | Medium | Tune (λ, γ) more carefully, or fall back to user-provided target descriptions |
| 3 — P2P compose with PEZ outputs | Medium | Investigate why; may require token-order regularization in PEZ-1/PEZ-2 |
| 4 — Useful Pareto for refinement ε | Low | Operate at ε=0 (pure discrete); main architecture still works |

The project's contribution lives or dies on **sub-claims 2 and 3**.
Sub-claim 1 is supporting (better source representation); sub-claim 4
is a fidelity boost that's nice to have but not essential.

- Sub-claim 1 fails → caption-source + PEZ-2 still gives instruction-
  conditioned editing. Smaller contribution but still novel.
- Sub-claim 2 fails → no automated instruction following; user has to
  manually describe target prompts. Project pivots to "PEZ as
  source representation for attention editing" — Option 1 from our
  earlier discussion.
- Sub-claim 3 fails → core architecture broken; investigate token-
  order regularization or per-position constraints in PEZ.
- Sub-claim 4 fails → drop refinement; ε=0 throughout. No fidelity
  boost but everything else works.

### What success enables

If all four sub-claims hold, the editing pipeline becomes:

```
Per edit:
  1. PEZ-1 on source image (cached) → source prompt
  2. PEZ-2 on (source image, user instruction), warm-started from
     PEZ-1 → target prompt
  3. (Optional) Bounded refinement on PEZ-1/PEZ-2 outputs
  4. DDIM-invert source under source prompt encoding
  5. Run P2P edit:
     - Token-ID matching aligns shared positions
     - Cross-attention swap copies source attention at mapped positions
     - Differing positions are unmapped → drive local blend mask
  6. Decode → edited image
```

The whole pipeline is built from one PEZ algorithm (run twice with
different conditioning), the existing inversion code, and the
existing P2P machinery. **No new attention controllers, no LLM,
no rule-based parser, no learned task-specific model.** This is the
demonstration in Section 11.

## 3. The core method

The architecture has three core pieces:
- **3.1**: PEZ-1 — source-image inversion to discrete prompt
- **3.2**: PEZ-2 — instruction-conditioned target generation
- **3.3**: P2P integration

Plus two supporting subsections:
- **3.4**: Bounded continuous refinement (optional fidelity lever)
- **3.5**: Why this fixes additive editing structurally

And one summary subsection:
- **3.6**: Computational requirements
- **3.7**: End-to-end pipeline recipe

Each maps to one or more of the sub-claims in Section 2.

### 3.1 PEZ on the source image (sub-claim 1)

**What it does.** Replaces BLIP-2 captioning with discrete prompt search
to produce a source prompt directly optimized for *recovering the image
through our pipeline*, not for from-scratch generation.

**Algorithm.** PEZ (Wen et al. 2023) maintains a soft prompt — N
continuous embedding vectors — and runs gradient updates with a
straight-through projection to the nearest CLIP vocabulary tokens.
We adapt the algorithm but **modify the loss** so that the prompt is
optimized for our actual reconstruction pipeline rather than for
matching CLIP-image similarity (which would be vanilla PEZ).

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

**What it does.** Runs PEZ a second time with two simultaneous CLIP
similarity targets — the source image and the user's instruction text
— producing a target prompt that satisfies both.

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
soft_prompt = init_from_pez1_tokens()                # [N, 768]  (warm start)

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

# Backprop through CLIP/U-Net; project to discrete vocab; repeat.
```

Three loss terms, three roles:

1. **Source preservation** (`L_source`): SDS-CFG surrogate for
   "prompt + null-text reconstruct the source image through our
   pipeline." Same loss family as PEZ-1; same null-text used.
2. **Instruction following** (`L_instr`): pulls the prompt toward
   describing the desired post-edit outcome. The instruction is text;
   text-text CLIP similarity is the appropriate signal.
3. **Warm-start anchor** (`L_anchor`): keeps the soft prompt close
   to PEZ-1's discrete tokens unless an instruction pressure overrides.
   This is what gives us the "minimal edit" property — most positions
   stay at PEZ-1's tokens; only positions under strong instruction
   pressure shift.

**Why this is not the same as classical instruction-following.**
CLIP doesn't really understand instructions like
`"change the animal into a cat"` as actions. It encodes them as
descriptions, and the encoding is dominated by the *target / changed
content* (`cat`) rather than by action verbs (`change`).

The PEZ-2 optimization exploits this: it finds a discrete prompt that
matches both the source image and the (target-content-dominated)
instruction. The result naturally describes the source's composition
with the changed content swapped in. There's no parsing of the
instruction's structure required — the semantic content of the
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
| Soft prompt | `[N, 768]` continuous vectors | The optimization variable; gradients land here |
| Pooled / projected (joint space) | `[768]` | Where similarity losses are computed |
| Discrete tokens | `[N]` token IDs | Final output via projection |

The soft prompt is initialized from the embeddings of PEZ-1's
discovered token IDs (warm start). The CLIP transformer contextualizes
the soft prompt, pools at EOS, and projects into joint space. The
losses compare in joint space (where CLIP was trained). The discrete
projection step (snap to nearest vocab embedding, with straight-
through gradient) happens at each step or every k steps.

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

**What PEZ-2 outputs in our running examples:**

For source image of a brown dog and instruction `"change the animal
into a cat"`:
```
PEZ-1 output:  [a, photo, of, a, fluffy, brown, dog, sitting, grass]
PEZ-2 output:  [a, photo, of, a, fluffy, brown, cat, sitting, grass]
                                         ^^^
                            position 6 changed; rest preserved
```

For instruction `"add a bowtie"`:
```
PEZ-1 output:  [a, photo, of, a, fluffy, brown, dog, sitting, grass]
PEZ-2 output:  [a, photo, of, a, fluffy, brown, dog, with, a, bowtie]
                                                  ^^^^^^^^^^^^^^^^^^^
                                          end positions changed/added
```

For instruction `"turn its fur brown"` (suppose source had a black dog):
```
PEZ-1 output:  [a, photo, of, a, fluffy, black, dog, sitting, grass]
PEZ-2 output:  [a, photo, of, a, fluffy, brown, dog, sitting, grass]
                                         ^^^^^
                              attribute swapped; rest preserved
```

These are illustrative — actual PEZ-2 outputs will depend on the
optimization's convergence behavior, which R2 characterizes.

**The detail-richness advantage at large N.** A subtle but important
property of this formulation: with a long prompt budget (large N),
PEZ-2 automatically discovers visual properties of the desired edit
that a user would not think to specify in a hand-crafted prompt.

Consider an example:

```
Source image:    A photo of a black husky with thick fur on a wooden table.

PEZ-1 (large N): "a high resolution photograph of a black husky with
                  thick fur sitting on a wooden table under soft natural
                  lighting"

User instruction: "change the dog into a cat"

PEZ-2 output:    "a high resolution photograph of a black bombay cat
                  with long fur sitting on a wooden table under soft
                  natural lighting"
```

The breed `bombay` (a black, fluffy cat) is not something the user
typed — they only said `"a cat"`. PEZ-2's joint optimization finds it
automatically because:

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

**No modifications required.** This is a positive claim: existing
P2P machinery composes with the PEZ-1 / PEZ-2 prompts unchanged.
We verify this rather than building anything new on the attention-
editing side.

**Why it works.**

- **Most positions match between PEZ-1 and PEZ-2 outputs** because
  PEZ-2 is warm-started from PEZ-1 with a regularizer keeping it
  close. Token-ID matching (used by LCS alignment in
  [cross_attention.py:64-80](attention_control/cross_attention.py#L64-L80))
  trivially maps shared positions.
- **Positions where PEZ-1 and PEZ-2 differ are the edits** — they're
  unmapped by P2P alignment. P2P leaves these positions alone (no
  injection); they're free to attend organically to whatever the new
  tokens semantically pull toward.
- **The local-blend mask is built from cross-attention to the
  unmapped positions.** The mask spatially localizes where the edit
  should appear; outside the mask, source structure is preserved via
  P2P injection.

**Refinements (Section 3.4) live at the input to CLIP.** Δ_i
perturbations are applied when looking up token embeddings; the
resulting input sequence is fed to CLIP's transformer, producing the
77×768 contextual encoding. P2P operates on this contextual encoding
without needing to know that some embeddings were refined.

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
2. It blends globally — no spatial localization. Local blend (which
   we use) provides the spatial mechanism this approach lacks.
3. SD has one generator. "Experts" in our setting are conditioning
   regions in the 77×768 prompt tensor, not separate models. Our
   P2P-based setup IS an MoE — at the cross-attention level, with
   token positions as experts and local blend as spatial gating —
   matching SD's architecture natively.

### 3.4 Bounded continuous refinement (sub-claim 4)

**What it does.** Recovers fidelity beyond PEZ's discrete vocabulary
ceiling without losing position-stability.

**Mechanism.** For each PEZ-discovered token at position `i`, learn
a perturbation `Δ_i ∈ ℝ^768` such that the refined embedding
`e_{w_i} + Δ_i` better reconstructs the training image. Constrain
`||Δ_i|| ≤ ε` either via hard projection after each gradient step or
via a Lagrangian penalty `λ · ||Δ_i||²`.

**Where refinement applies.**
- **PEZ-1 outputs.** Refine source tokens to improve inversion
  fidelity. Budget `ε_source` (e.g., 0.1).
- **PEZ-2 outputs.** Refine target tokens for edit fidelity.
  Budget `ε_target` (e.g., 0.1, possibly different from ε_source).

**Where refined embeddings get applied at runtime.** At the input to
CLIP's text encoder. Tokenize the prompt normally, look up each
token's vocabulary embedding, then for tokens with refinements, add
the Δ_i to the lookup. The resulting input embedding sequence (77 ×
768) is fed to CLIP's transformer normally. The 77×768 contextual
encoding it produces is what P2P sees.

**The Pareto frontier.**
- `ε = 0`: pure discrete. Position-stability automatic. Fidelity
  bounded by vocabulary expressiveness.
- `ε → ∞`: continuous textual inversion (in the limit). High
  fidelity. Position-stability collapses (smearing returns).
- Useful operating point: ε small enough that contextual-encoding
  footprint stays concentrated at each token's position, but large
  enough that fidelity meaningfully improves.

Sub-claim 4's empirical question is whether such operating points
exist for our use cases. The architecture works at ε=0 (no
refinement) so this is a fidelity boost, not a load-bearing piece.

### 3.5 Why this fixes additive editing

The two failure modes from Section 1's "Additive editing is the
hardest case" subsection are addressed structurally:

**Failure mode 1 (local-blend mask collapse) — fixed because:** PEZ-1
and PEZ-2 outputs are real CLIP vocabulary tokens. CLIP was trained
on captions; cross-attention to a vocabulary token at position `k` is
what the model is built to handle, and the column is naturally
spatially localized. The local-blend mask derived from this column
inherits that locality.

**Failure mode 2 (P2P injection erasure) — fixed because:** PEZ-2's
warm-start property means most positions in source and target are
identical. Edits live at the positions where they differ — and at
those positions, P2P sees mismatched token IDs and treats them as
unmapped. Unmapped means P2P does NOT inject source content onto
target's column. The new content stays at its position, untouched
by the alignment mechanism.

For substitution edits (`dog → cat` swap):
- Most source/target positions have matching tokens → P2P injects.
- The swapped position has different tokens → unmapped → P2P leaves
  alone → cat's organic cross-attention renders.

For additive edits (`+ wearing a bowtie`):
- Source positions stay matched to target's leading positions →
  P2P injects.
- Target's trailing positions don't exist in source → unmapped → free
  to render new content.

In both cases, the architecture's correctness comes from the warm-
start property keeping PEZ-2 close to PEZ-1 except where the
instruction demands change.

### 3.6 Computational requirements summary

No model training. The base models (CLIP, SD U-Net, VAE) stay frozen
throughout. Three optimization procedures:

| Operation | Cost | Frequency |
|---|---|---|
| PEZ-1 on a source image | ~15–30 min GPU | Per source image (cached) |
| PEZ-2 (instruction-conditioned, warm-started) | ~5–10 min GPU | Per (image, instruction) pair |
| Δ refinement (PEZ-1 and/or PEZ-2) | ~5–20 min GPU | Per refinement target (optional) |

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
     image → discrete source tokens [w_s1, ..., w_sN]
     Optional: bounded refinement → Δ_s1, ..., Δ_sN, ||Δ|| ≤ ε_source

  2. PEZ-2 on (source image, instruction), warm-started from PEZ-1:
     - Initialize soft prompt at PEZ-1's vocabulary embeddings
     - Optimize with joint loss:
         L = -cos_sim(prompt_emb, image_emb)
             -λ * cos_sim(prompt_emb, instruction_emb)
             +γ * ||soft_prompt - soft_prompt_init||²
     - Project to discrete tokens at convergence
     → discrete target tokens [w_t1, ..., w_tM]
       (M ≈ N for substitutions; M slightly > N for additions)
     Optional: bounded refinement → Δ_t1, ..., Δ_tM, ||Δ|| ≤ ε_target

  3. Encode both prompts through CLIP:
     For each prompt:
       - Tokenize → token IDs
       - Look up vocabulary embeddings → input 77×768
       - Add Δ_i at refined positions (if any)
       - Run CLIP text-encoder transformer → 77×768 contextual encoding

  4. DDIM-invert source image under source contextual encoding
     (use existing src/inversion.py)

  5. Compute token mapping between source and target prompts:
     - Positions where token IDs match → mapped (1:1)
     - Positions where token IDs differ → unmapped (the edits)
     - Positions in target only → unmapped (additions)

  6. Run editing denoising loop with [source, target] batch:
     - CrossAttentionController copies source attention columns to
       target at mapped positions
     - LocalBlend uses target_token_indices = unmapped positions to
       build a spatial mask; injection is gated by the mask outside
       its boundary

  7. Decode final latent → edited image
```

The crucial property: **all existing attention-editing machinery
works unchanged.** PEZ-2's warm-start ensures source and target share
most token IDs at the same positions; P2P alignment via token-ID
matching is therefore trivial. The positions that differ are exactly
the edit regions, and they're naturally identified as unmapped — which
is what local blend needs to localize the edit.

No new attention controllers, no modifications to the existing
denoising loop.

## 4. Evaluation plan

### 4.1 Source-image inversion quality (sub-claim 1)

For a held-out set of 20 source images, measure inversion+reconstruction
fidelity under three source-prompt strategies:

- **BLIP-2 caption**: as in the v1 plan. Caption then DDIM-invert.
- **Hand-crafted caption**: a human writes a careful description.
  Establishes a ceiling for caption-based methods.
- **PEZ-discovered prompt**: the proposed source representation.
- **PEZ + refined source tokens** at multiple ε ∈ {0.05, 0.1, 0.2}.

Metrics: PSNR / SSIM / LPIPS on `recon = decode(reconstruct(invert(image,
prompt)))`. Plus footprint concentration of refined source tokens to
verify ε bounds are doing what we expect.

### 4.2 PEZ-2 target prompt quality (sub-claim 2) — Knob 1: divergence

This evaluation measures **Knob 1**: how far PEZ-2's prompt diverges
from PEZ-1, controlled by `(λ_instruction, γ_anchor)`. The sweep
characterizes the operating range of Knob 1 independent of editing-
time settings.

For a fixed test set of 30 (source image, instruction) pairs covering
substitution, addition, and attribute change, measure:

- **Token preservation rate**: fraction of PEZ-1's tokens that appear
  at the same position in PEZ-2's output. Higher = warm-start working.
  Target: ≥70% for substitutions, ≥85% for additions (where most
  positions are unchanged).
- **Edit semantic correctness** (human raters, paired comparisons):
  does PEZ-2's output prompt describe an image satisfying the
  instruction?
- **CLIP similarity** between PEZ-2's output and a hand-crafted
  ground-truth target prompt for each test case.
- **(λ_instruction, γ_anchor) ablation**: sweep `λ_instruction ∈ {0.5,
  1.0, 2.0, 5.0}` and `γ_anchor ∈ {0.01, 0.1, 1.0}`. Report
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
PEZ-2 producing target prompts whose token structure doesn't align
cleanly under LCS — investigate.

### 4.4 End-to-end editing quality (sub-claim 4) — Knob 2: edit aggressiveness

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
- **Continuous TI + P2P (broken baseline)**: continuous textual
  inversion injected at target positions.
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

- Proposed method at moderate Knob-2 beats continuous TI + P2P
  decisively on additive edits.
- Proposed method matches or beats InstructPix2Pix on structure
  preservation. Concept fidelity should be comparable.
- Proposed method approaches the BLIP-2 + hand-crafted ceiling on edit
  quality.
- Knob 2 has predictable trade-offs: subtle → high structure
  preservation but low concept fidelity; loud → high concept fidelity
  but degraded structure preservation. Moderate sits at the elbow.

### 4.5 Knob 1 × Knob 2 interaction grid

The two control knobs identified above operate at different stages of
the pipeline and are nominally independent, but interact in their
effect on the final edit. This evaluation maps the joint operating
space.

**Why this is cheap.** Knob 1 lives at PEZ-2 optimization time
(needs to re-run PEZ-2: ~10 min per setting). Knob 2 lives at editing
time (just re-run the denoising loop: ~2 min per setting). For each
PEZ-2 output, all Knob-2 settings can be evaluated for ~2 min × |Knob
2| extra time — no re-optimization needed.

**Sweep design.**

For each of 5 representative (image, instruction) pairs in the test
set, evaluate the 3 × 3 = 9 setting grid:

```
                     Knob 2 (edit aggressiveness)
                     Subtle    Moderate    Loud
Knob 1     Conservative   ⬜       ⬜         ⬜
(divergence)  Moderate    ⬜       ⬜         ⬜
              Aggressive  ⬜       ⬜         ⬜
```

**Cost per (image, instruction) pair:**
- 3 PEZ-2 runs × ~10 min = ~30 min
- 9 edits × ~2 min = ~18 min
- Total: ~50 min per pair

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

### 4.7 P2P alignment robustness on PEZ-derived prompts

PEZ-1 and PEZ-2 may produce token sequences with unusual orderings.
Verify that P2P's LCS alignment still produces sensible mappings:

- Source is PEZ-1-discovered (unusual token order)
- Target is PEZ-2-discovered, warm-started from PEZ-1 (mostly aligned
  but some positions differ)
- Multi-attribute edits where multiple positions differ between source
  and target

Failure mode: PEZ chooses common stop-words at multiple positions,
causing LCS to align them ambiguously. If observed, replace LCS with
a CLIP-embedding-based semantic aligner (see Appendix B).

## 5. Codebase organization

New code lives under `src/pez/`, `src/refinement/`, `src/splice/`.
Existing code (`src/inversion.py`, `attention_control/`,
`src/utils.py`) is reused unchanged.

Proposed layout:

```
src/
  pez/
    __init__.py
    search.py             # core PEZ algorithm: soft prompt + projection
                          # used by both source_inversion and instruction_conditioned
    source_inversion.py   # PEZ-1: image → discrete source prompt
                          # (caches outputs per image hash)
    instruction_conditioned.py
                          # PEZ-2: (image, instruction) → discrete target prompt
                          # warm-started from PEZ-1, with joint loss + γ regularizer

  refinement/
    __init__.py
    delta.py              # learnable Δ_i perturbation per token
    train.py              # bounded refinement training loop
                          # (Lagrangian or hard-projection ε constraint)
                          # applies to both PEZ-1 and PEZ-2 outputs

  splice/
    __init__.py
    encode_with_refinement.py
                          # tokenize a prompt, look up embeddings,
                          # add Δ at refined positions, run CLIP transformer
    align.py              # token-ID matching between source and target prompts;
                          # returns (mapping, unmapped_target_indices)

  metrics/
    __init__.py
    fidelity.py           # PSNR / SSIM / LPIPS of inversion+reconstruction
    footprint.py          # contextual-encoding footprint concentration
    edit_quality.py       # structure preservation, concept fidelity (CLIP),
                          # mask quality (IoU vs ground-truth edit region)

attention_control/
  local_blend.py          # built per Appendix A (needed by R4)

notebooks/
  R1_pez_source.ipynb        # PEZ-1 vs. BLIP-2 captioning fidelity comparison
  R2_pez_instruction.ipynb   # PEZ-2 ablation of (λ, γ); token preservation
                             # rate; edit semantic correctness
  R3_refinement_pareto.ipynb # bounded refinement ε ablation
  R4_full_evaluation.ipynb   # end-to-end editing comparison vs.
                             # IP2P, continuous TI, caption baselines

data/
  test_images/               # held-out source images for editing
    person_at_desk.png
    dog_on_grass.png
    ...
  test_instructions.json     # paired instructions for each test image
  ground_truth_target_prompts.json
                             # human-written target prompts for evaluation

cache/
  pez_source_prompts/        # cached PEZ-1 outputs per source image
                             # (each PEZ-1 run is ~15-30min; cache aggressively)
  pez_target_prompts/        # cached PEZ-2 outputs per (image, instruction)
                             # (each PEZ-2 run is ~5-10min; cache when possible)
  refined_deltas/            # cached Δ refinements
```

The three library directories (`pez/`, `refinement/`, `splice/`) form
a clean separation: PEZ produces discrete token sequences (used both
for source inversion and instruction-conditioned target generation),
refinement produces optional small perturbations, splice assembles
prompts and aligns them for P2P. Each is independently testable.

## 6. Existing-code reuse map

Existing code in the repo is reused unchanged except for adding the
LocalBlend module needed by R4.

| Component | Status | When needed |
|---|---|---|
| Existing inversion (DDIM + null-text) in `src/inversion.py` | Required, unchanged | All phases that involve image reconstruction. |
| Existing attention controllers (P2P) in `attention_control/` | Required, unchanged | Phase R4. The architectural claim is that these continue to work unchanged when fed PEZ-derived prompts. |
| Existing SD utility code in `src/utils.py` | Required, unchanged | All phases. |
| New: `attention_control/local_blend.py` | Required (built when starting R4) | Phase R4. Specification in Appendix A. |
| New: `src/semantic_alignment.py` | Optional fallback for R4 | Only if LCS alignment proves insufficient for PEZ outputs. Specification in Appendix B. |
| BLIP-2 captioning wrapper | Used as R1 baseline only | A minimal BLIP-2 wrapper for the source-fidelity comparison in R1. Not part of the proposed pipeline; exists only for the comparison. |

The research never modifies existing code; it only adds new modules.

## 7. Phased experimental milestones

The project breaks into four phases, each producing a discrete artifact.

### Phase R1 — PEZ on the source image (sub-claim 1)

**Goal:** a working PEZ implementation that takes a source image and
produces a discrete CLIP-vocabulary prompt, plus measurements showing
PEZ-discovered prompts give higher inversion fidelity than BLIP-2
captions.

**Files to create:**

- `src/pez/search.py`
  ```python
  def pez_search(
      target: Image.Image | list[Image.Image],
      clip_model,
      tokenizer,
      prompt_length: int = 8,
      num_steps: int = 1500,
      lr: float = 0.1,
      device: torch.device,
  ) -> list[int]:
      """Run PEZ optimization (Wen et al. 2023). Returns CLIP vocab
      token IDs that maximize CLIP-image similarity to the target.

      Algorithm:
        - Initialize a [prompt_length, 768] soft prompt randomly.
        - For num_steps:
          - Project soft prompt to nearest vocabulary tokens (hard).
          - Compute CLIP loss between soft prompt encoding and target
            image embedding.
          - Backprop through projection (straight-through estimator).
          - Update soft prompt via Adam.
      """
  ```
- `src/pez/source_inversion.py`
  ```python
  def pez_invert_source(
      image: Image.Image,
      clip_model,
      tokenizer,
      prompt_length: int = 8,
      num_steps: int = 1500,
      cache_dir: Path | None = None,
  ) -> str:
      """PEZ on a single source image. Returns a string prompt
      decoded from discovered token IDs. Caches to disk by image hash.
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

- A working PEZ implementation, validated on a few test images.
- 10 source images with cached PEZ prompts in
  `cache/pez_source_prompts/`.
- A comparison table per source image:

  | source-prompt strategy | PSNR | SSIM | LPIPS |
  |---|---|---|---|
  | BLIP-2 caption | ... | ... | ... |
  | Hand-crafted caption (ceiling) | ... | ... | ... |
  | PEZ-discovered prompt | ... | ... | ... |

**Expected finding:**

PEZ-discovered prompts give comparable or better inversion fidelity
than BLIP-2 captions, particularly on images with hard-to-describe
content (specific patterns, unusual compositions, fine textures).
Hand-crafted captions remain the ceiling but require human effort.

If PEZ doesn't beat BLIP-2 (sub-claim 1 fails), pivot: fall back to
BLIP-2 captioning for source representation. PEZ-2 (R2 onward) still
works on top of caption-derived sources — it just becomes "PEZ-2 from
a caption" instead of "PEZ-2 from PEZ-1". The architecture survives;
the source-representation contribution is reduced.

Estimated time: 1-2 weeks. Mostly adapting Wen et al.'s PEZ codebase
to integrate with the existing inversion pipeline.

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
      pez1_tokens: list[int],          # warm-start from PEZ-1
      clip_model,
      tokenizer,
      lambda_instr: float = 1.0,        # instruction strength
      gamma: float = 0.1,               # warm-start anchor
      num_steps: int = 500,
  ) -> list[int]:
      """PEZ-2: optimize a soft prompt warm-started from pez1_tokens
      with joint CLIP-image and CLIP-instruction similarity losses,
      plus an L2 regularizer pulling the soft prompt back toward its
      initialization.

      Loss:
        L = -cos_sim(prompt_emb, image_emb)
            -lambda_instr * cos_sim(prompt_emb, instr_emb)
            +gamma * ||soft_prompt - soft_prompt_init||^2

      Returns the discrete token IDs after final projection."""
  ```
- `src/splice/align.py`
  ```python
  def align_pez_prompts(
      source_token_ids: list[int],
      target_token_ids: list[int],
  ) -> tuple[dict[int, int], list[int]]:
      """LCS-based alignment between source and target token sequences.

      Returns:
        - mapping: {source_pos: target_pos} for matched token IDs
        - unmapped_target_indices: target positions with no source match
          (these drive local blend in editing)"""
  ```

**What to produce:**

- A working PEZ-2 implementation, validated on 5 hand-picked
  (source_image, instruction) test cases with known expected target
  prompts.
- A (λ, γ) ablation grid: for each test case, sweep λ ∈ {0.5, 1.0,
  2.0, 5.0} and γ ∈ {0.01, 0.1, 1.0}. Report:
  - Token preservation rate (PEZ-1 vs. PEZ-2 token-ID overlap)
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

### Phase R3 — Bounded continuous refinement (sub-claim 4)

**Goal:** add the continuous refinement layer; characterize the Pareto
frontier between fidelity and position-stability for both PEZ-1 and
PEZ-2 outputs.

**Files to create:**

- `src/refinement/delta.py`
  ```python
  class TokenRefinements:
      """Manages a list of [768] perturbation tensors, one per refined
      token position.

      Provides:
        - parameters() for optimizer (the Δ_i tensors)
        - apply(input_embeddings, positions): adds Δ_i at positions
          inside a [1, 77, 768] input embedding tensor
        - epsilon_clip(): hard-projects each Δ_i to ||Δ|| ≤ ε
          (alternative to a Lagrangian penalty)
      """
  ```
- `src/refinement/train.py`
  ```python
  def refine_embeddings(
      target_images: list[Image.Image],
      base_token_ids: list[int],
      pipeline,
      epsilon: float = 0.1,
      num_steps: int = 1000,
      lr: float = 1e-4,
      method: Literal["projection", "lagrangian"] = "projection",
  ) -> list[torch.Tensor]:
      """Train Δ_i for each token in base_token_ids such that
      e_{w_i} + Δ_i better reconstructs target_images, subject to
      ||Δ_i|| ≤ ε."""
  ```
- `src/splice/encode_with_refinement.py`
  ```python
  def encode_prompt_with_refinement(
      prompt: str,
      tokenizer,
      text_encoder,
      refinements: dict[int, torch.Tensor],   # {position: Δ}
  ) -> torch.Tensor:
      """Tokenize prompt; look up vocabulary embeddings; add Δ at
      specified positions; run CLIP transformer; return [1, 77, 768]
      contextual encoding.

      THE core integration point: where discrete tokens + bounded
      refinements meet the encoder pipeline."""
  ```

**What to produce:**

- A refinement implementation usable for both PEZ-1 outputs (source
  tokens) and PEZ-2 outputs (target tokens).
- Pareto curves: x = footprint concentration, y = reconstruction
  fidelity, one point per ε ∈ {0, 0.05, 0.1, 0.2, 0.5, 1.0}.
- Recommended operating ε, selected as the largest ε at which
  footprint concentration stays above 0.7 (or some chosen stability
  threshold).

**Expected finding:**

A useful Pareto frontier exists at ε ≈ 0.1 giving meaningful fidelity
improvement while keeping footprint concentration in the 0.7–0.9 range.

If no useful frontier exists (sub-claim 4 fails), operate at ε = 0
(pure discrete). Architecture still works, just with PEZ's known
fidelity ceiling. Refinement is optional, not load-bearing.

Estimated time: 2 weeks.

---

### Phase R4 — End-to-end editing evaluation (sub-claim 4 + paper)

**Goal:** end-to-end editing experiments with the full PEZ-1 + PEZ-2
+ P2P architecture vs. baselines; paper-grade writeup.

**Files to create:**

- `src/metrics/edit_quality.py` — implementations of metrics in
  Section 4.4: structural SSIM, concept CLIP similarity, identity
  LPIPS, mask IoU.
- `attention_control/local_blend.py` — built per Appendix A.

**Demo and evaluation notebook: `notebooks/R4_full_evaluation.ipynb`**

For each of 50 (source image, instruction) pairs:

1. Load cached PEZ-1 source prompt (from R1) and run PEZ-2 (from R2)
   to get target prompt. Apply optional refinements (from R3).
2. Encode source and target prompts through CLIP with refinements
   applied at input.
3. DDIM-invert source under source encoding.
4. Compute LCS alignment between source and target token IDs.
5. Set up controllers:
   - `CrossAttentionController` with the LCS mapping
   - `LocalBlend` with `target_token_indices` = unmapped target positions
6. Run editing denoising loop.
7. Save side-by-side comparisons:
   - Source image
   - Edit using **continuous TI + P2P** (broken baseline)
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
  - Without refinement (ε=0 everywhere)
- Qualitative figure for the paper.
- Paper draft: motivation, method, results, related work, limitations.

**Expected finding:**

Proposed architecture beats continuous-TI baseline decisively on
additive edits, matches/beats InstructPix2Pix on structure
preservation, and approaches the BLIP-2 + hand-crafted ceiling on edit
quality (showing PEZ-2 successfully automates the human's role).

Estimated time: 3–4 weeks including writeup iteration.

---

**Total estimated timeline: 7-9 weeks of focused work.**

A reasonable milestone cadence: R1 done by week 1, R2 by week 3, R3
by week 5, R4 by weeks 6-9.

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

5. **Vocabulary biasing during PEZ search.** Restricting PEZ's
   vocabulary to common edit-friendly words might help PEZ-2 produce
   more sensible target prompts (since the warm start is more natural).
   Worth ablating.

6. **SDS stability.** DreamFusion-style SDS has known issues
   (over-saturation, mode collapse, sensitivity to timestep sampling).
   For PEZ — which optimizes a finite vocabulary projection of a soft
   prompt rather than continuous pixels — these issues may manifest
   differently. Worth measuring and possibly mitigating with VSD
   (Variational Score Distillation, Wang et al. 2023) if vanilla
   SDS proves unstable.

7. **Refinement budget allocation.** Two ε values to tune — one for
   PEZ-1, one for PEZ-2. Should they be coupled, or independent?

8. **Interaction with P+ / per-layer textual inversion.** P+ (Voynov
   et al. 2023) shows per-layer embeddings capture concepts more
   richly. Could PEZ-1 / PEZ-2 extend to the per-layer setting? Likely
   yes but adaptation needed.

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

**Categories of instruction the system handles well:**
- **Substitution** — `"change X to Y"`, `"replace X with Y"`. CLIP
  encoding has Y as dominant content; PEZ-2 finds a token shifting
  X → Y at the appropriate prompt position.
- **Addition** — `"add a Y"`, `"with a Y"`, `"wearing a Y"`. Same
  mechanism; Y dominates.
- **Attribute change** — `"make it brown"`, `"turn its X red"`. The
  attribute dominates the encoding.
- **Style change** — `"make it black and white"`, `"in oil painting
  style"`. The style descriptor dominates.

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

### How users should phrase instructions

For best results, phrase instructions as **descriptions of the
desired image content**, not as imperatives or modal statements:

| Instead of | Phrase as |
|---|---|
| `"make the animal like eating fish more"` | `"add a fish in front of the animal"` (or accept the graceful degradation: `"animal eating fish"`) |
| `"make it not red"` | `"make it blue"` or whatever specific color is desired |
| `"make it bigger"` | `"a giant version of X"` if size relative to a referent is the intent |
| `"remove the dog"` | `"a photo of an empty backyard"` (i.e., describe the desired post-edit content directly) |

Users who phrase image-content instructions get strong PEZ-2
behavior. Users who phrase modal/intentional instructions get
graceful degradation to the closest image-content interpretation.
This is the expected operating envelope.

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
  Textual Inversion, identity-locking LoRAs); the discrete-token
  inversion in PEZ-1 doesn't carry per-instance identity strongly
  enough for this.

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
| **PEZ / Hard Prompts Made Easy** (Wen 2023) | Discrete prompt search for image-to-text inversion | Foundation we build on; not previously applied as a source-representation primitive for attention-based editing, nor extended to instruction-conditioned target generation |
| **Textual Inversion** (Gal 2022) | Learns single continuous-token concept embeddings | Continuous embeddings smear across contextual encodings, breaking P2P alignment; we use discrete tokens specifically to avoid this |
| **DreamBooth** (Ruiz 2022) | Fine-tunes the U-Net for a specific concept | Modifies model weights; concept not transportable via prompt manipulation |
| **InstructPix2Pix** (Brooks 2023) | Trains a diffusion model end-to-end for instruction-conditioned image editing | Replaces P2P entirely; requires bootstrapped training data via GPT-3; produces a black-box editing model rather than a compositional pipeline |
| **Composable Diffusion** (Liu 2022) | Combines multiple text conditionings via composed CFG | Generates from scratch — no source structure preservation. Wouldn't compose with our P2P setup without adding the same machinery anyway |
| **Custom Diffusion** (Kumari 2022) | Optimizes K/V projections per concept | Composition lives in projection space; doesn't compose with P2P's prompt-side column-swap mechanism |
| **P+** (Voynov 2023) | Per-layer textual inversion | Orthogonal — could combine with PEZ for richer per-layer discrete representations (Open Question 6) |
| **Concept Sliders** (Gandikota 2023) | Trains directional axes between concepts | Different abstraction (continuous attribute axes); uses LoRAs not prompts |
| **Mix-of-Show** (Gu 2023) | LoRA fusion across multiple concepts | LoRA-level composition, not prompt-level |
| **Prompt-to-Prompt** (Hertz 2022) | Cross-attention column swap for editing | Built for natural-language prompts; this proposal extends its applicability to PEZ-derived prompts via the discrete-token-anchor argument |
| **BLIP-2 captioning** (Li 2023) | Image-to-text via caption | Used as a baseline; PEZ-1 replaces it for source representation, with PEZ optimizing CLIP-image similarity directly |

The contribution is **the two-PEZ architecture for instruction-
conditioned editing** — PEZ-1 for source representation, PEZ-2 for
instruction-conditioned target generation, both composing with existing
P2P machinery unchanged. The instruction-following step is
embedding-space optimization (PEZ-2's joint loss) rather than parsing
or LLM-based reasoning. No cited method does this — each solves a
related but distinct piece.

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
   - **PEZ-1** runs on the source image with N=15 → discrete source
     tokens that capture the source's visual details, e.g.
     `["a", "high", "resolution", "photograph", "of", "a", "black",
     "husky", "with", "thick", "fur", "on", "a", "wooden", "table"]`.
   - **PEZ-2** runs on `(source image, instruction)`, warm-started
     from PEZ-1's tokens, with the joint loss:
     ```
     L = -cos_sim(prompt, image) - λ·cos_sim(prompt, instruction)
         + γ·||soft_prompt - init||²
     ```
     The instruction's CLIP encoding is dominated by `"cat"`. The
     joint optimization finds tokens that satisfy:
     - Most positions stay at PEZ-1's tokens (warm-start anchor)
     - Some position shifts toward "cat" (instruction term)
     - The chosen cat must visually match the source — black, thick
       fur, large breed (source-similarity term)
     
     Result: `["a", "high", "resolution", "photograph", "of", "a",
     "black", "bombay", "with", "long", "fur", "on", "a", "wooden",
     "table"]`. The breed `bombay` (a black, fluffy cat) was *not*
     in the user's instruction. PEZ-2 found it by satisfying source
     visual properties simultaneously with instruction semantics.
   - DDIM-inverts the photo under PEZ-1's source encoding.
   - Runs P2P edit. Token-ID matching aligns most positions; 
     the differing positions (`husky → bombay`, `thick → long`) are
     unmapped → drive the local-blend mask. P2P injects at unchanged
     positions, preserving the source's structural skeleton.
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
section 7. The repo doesn't currently have a PEZ implementation;
adapting one (from Wen et al.'s public codebase, into `src/pez/`
against the existing SD2.1 + frozen-component setup in `src/utils.py`)
is the entry point. Once PEZ is producing discrete prompts that
work with the existing DDIM inversion pipeline, the rest of the
project follows naturally.

The first concrete commit should:

1. Create `src/pez/` with module stubs.
2. Adapt PEZ from Wen et al.'s repo into `src/pez/search.py`. Verify
   it runs and produces sensible discrete token IDs against a single
   source image.
3. Wire up `src/pez/source_inversion.py` with disk caching, since each
   PEZ run is ~15-30 min and we want to iterate fast on downstream
   code.
4. Run PEZ on one test image; pass the resulting prompt through
   existing `src/inversion.py` to verify the integration. Compare
   reconstruction PSNR against a BLIP-2 caption baseline.

That's the smallest viable starting point. From here, the comparison
table for sub-claim 1 can be filled in, and Phase R2 (PEZ-2
instruction-conditioned generation) becomes a natural next step using
the same PEZ infrastructure with an additional loss term.

---

## Appendix A — LocalBlend specification

Required by Phase R4. Provides spatial gating so attention injection
can be disabled inside an edit region (giving the new content room
to render) and enabled outside (preserving source structure).

### Mechanism

At each denoising step, build a binary mask over the image's spatial
grid by aggregating cross-attention to the target prompt's "edit"
positions (the positions that differ between source and target — i.e.,
the unmapped target indices from LCS alignment).

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

## Appendix B — Semantic alignment fallback

Required only if Section 4.6's evaluation shows that LCS alignment on
PEZ-derived token IDs produces noisy mappings. Default is LCS; this
appendix is a fallback.

### Mechanism

Replace LCS with bipartite matching over CLIP-embedding cosine
similarity between source and target token contextual encodings:

1. Tokenize source and target prompts. Track which positions are real
   content tokens (skip BOS, EOS, padding).
2. Run CLIP text encoder. Take per-token contextual encodings (the
   77×768 unpooled output).
3. Cosine-similarity matrix `C[i, j]` between source position `i` and
   target position `j`, over content positions only.
4. Solve maximum-weight bipartite matching (Hungarian algorithm,
   `scipy.optimize.linear_sum_assignment`).
5. Filter assignments below `similarity_threshold` (default 0.7).
   These positions are "non-matches" and feed into LocalBlend.

### API

```python
def compute_semantic_token_mapping(
    text_encoder,
    tokenizer,
    source_prompt: str,
    target_prompt: str,
    device: torch.device,
    similarity_threshold: float = 0.7,
    exclude_special_tokens: bool = True,
) -> tuple[dict[int, int], list[int]]:
    """Returns:
      - mapping: {source_idx: target_idx} for matched tokens
      - unmapped_target_indices: target positions with no source match
        (used as LocalBlend's target_token_indices)
    """
```

### When to use

- LCS alignment in `attention_control/cross_attention.py:64-80` is
  the default. PEZ-derived prompts mostly share tokens at the same
  positions (warm-start property), so LCS works trivially in
  practice.
- Use semantic alignment only if PEZ-2's outputs occasionally produce
  reordered tokens or near-synonyms (e.g., PEZ-1 has `dog` and PEZ-2
  has `puppy` — token-ID mismatch but semantically the same).
  Empirical question for R4.
