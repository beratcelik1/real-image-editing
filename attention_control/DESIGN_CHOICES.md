# Cross-Attention Controller — Design Choices

Open decisions that must be resolved before implementing `cross_attention.py`.

---

## 1. Layer Identification

How are target layers specified to `register_attention_control`?

| Option | Example | Trade-off |
|---|---|---|
| Integer index | `{6, 7, 8}` | Simple, brittle across model variants |
| Layer name | `{"up_blocks.1.attentions.0"}` | Explicit, model-specific strings |
| Predicate function | `lambda name: "up" in name` | Flexible, requires caller knowledge |

**Decision needed:** Which identifier scheme to use, and whether it should be enforced at registration time or lazily at forward time.

---

## 2. Controller Interface — Callable vs. Stateful Class

**Option A — Callable**
```python
controller(attn_weights, is_cross, place_in_unet) -> attn_weights
```
Stateless per-call. Simple to compose. Cannot track injection history or blend across timesteps without external state.

**Option B — Stateful class**
```python
class AttentionController:
    def forward(self, attn_weights, is_cross, place_in_unet) -> attn_weights: ...
    def reset(self): ...  # called between diffusion steps
    def between_steps(self): ...  # optional hook after each step
```
Matches the original P2P paper implementation. Supports step counters, attention map storage, and cross-step blending. More boilerplate.

**Decision needed:** Whether the project needs cross-step state (e.g. averaging maps over timesteps). If yes, use Option B.

---

## 3. `place_in_unet` Granularity

**Coarse (3 values):** `"down"`, `"mid"`, `"up"`
- Sufficient for most P2P use cases.

**Fine-grained:** `"down_0"`, `"down_1"`, `"up_0"`, `"up_1"`, etc.
- Required if different blocks within a stage need different injection behavior.

**Decision needed:** Whether per-block control is needed. Coarse is easier to implement; fine-grained is more flexible.

---

## 4. Self-Attention Passthrough

The current design assumes `is_cross=False` calls are passed through unmodified.

- **Keep passthrough:** Simpler. Sufficient for text-driven edits that only modify cross-attention.
- **Allow self-attention injection:** Needed for structure/layout transfer between source and target (as used in P2P's attention re-injection for geometry preservation).

**Decision needed:** Whether layout/structure transfer is in scope. If yes, the controller must also handle `is_cross=False`.

---

## 5. Injection Timing Enforcement

The docstring recommends restricting injection to `"up"` layers for style-preserving edits, but does not enforce it.

- **Caller-enforced:** The controller decides which layers to modify based on `place_in_unet`. Flexible, but easy to misconfigure.
- **Framework-enforced:** `register_attention_control` accepts an explicit `inject_layers` allowlist and skips calls outside it before reaching the controller.

**Decision needed:** Whether the registration function should guard injection scope, or leave that responsibility to the controller implementation.
