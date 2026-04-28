"""LocalBlend — spatial gating for cross-attention injection during P2P edits.

A mask state that ``CrossAttentionController`` consults to decide
where to inject (structural preservation, outside the mask) vs.
where to leave the target alone (new content, inside the mask).

The mask used at denoising step ``t`` is computed from cross-attention
maps recorded at step ``t-1`` (one-step lag — within a denoising
step, cross-attention is recorded but the mask used by the same step's
injection comes from the previous step). The first denoising step has
no mask; injection is unmasked at step 0.

Specification: RESEARCH_PROPOSAL.md Appendix A and the design choice
in docs/p2p_pnp/DESIGN_CHOICES.md #1.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class LocalBlend:
    """Mask state for P2P cross-attention gating.

    Attached to ``CrossAttentionController`` via its ``local_blend``
    parameter. The controller feeds this object via
    ``record_cross_attention`` during its ``__call__`` and consults
    ``get_mask`` in ``_word_swap``; ``step()`` is called from the
    controller's ``step()`` to advance per-denoising-step state.

    Parameters
    ----------
    target_token_indices
        Positions in the target prompt's 77-token sequence whose
        cross-attention is used to build the mask. Typically these are
        the unmapped target tokens — the new words / changes that
        should drive the spatial mask. Provide via
        ``align_pez_prompts``'s ``unmapped_target_indices`` return.
    threshold
        Probability threshold (after max-normalization) above which a
        patch is considered "inside" the mask. Default 0.3 of max
        activation.
    base_resolution
        Side length at which the mask is computed before resampling.
        For SD2.1 at 768×768 the cross-attention layers come at 8/16/
        32/64 spatial resolutions; we pick one canonical (16 by
        default) to aggregate at, then resample on demand to whichever
        layer queries the mask.
    dilate_iters
        Optional morphological dilation (max-pool) applied to soften
        the mask boundary. 0 disables; 1 is the default.
    """

    def __init__(
        self,
        target_token_indices: list[int],
        threshold: float = 0.3,
        base_resolution: int = 16,
        dilate_iters: int = 1,
    ) -> None:
        if not target_token_indices:
            raise ValueError(
                "target_token_indices must be non-empty. "
                "If you have no unmapped positions, you don't need a "
                "LocalBlend (the controllers behave identically when "
                "local_blend=None)."
            )
        self.target_token_indices = list(target_token_indices)
        self.threshold = threshold
        self.base_resolution = base_resolution
        self.dilate_iters = dilate_iters

        # Mask used at the CURRENT denoising step. None at step 0.
        self._current_mask: torch.Tensor | None = None
        # Accumulator for the NEXT step's mask (filled by
        # record_cross_attention during this step's denoising).
        self._accumulator_sum: torch.Tensor | None = None
        self._accumulator_count: int = 0
        # Idempotency guard for step().
        self._finalized_step_id: int = -1
        # Number of denoising steps observed (for the guard).
        self._cur_step: int = 0

    # ------------------------------------------------------------------
    # Inputs from CrossAttentionController
    # ------------------------------------------------------------------

    def record_cross_attention(self, attn_4d: torch.Tensor) -> None:
        """Accumulate the target half's attention to selected token positions.

        Parameters
        ----------
        attn_4d : Tensor[batch, heads, spatial, tokens]
            Cross-attention probabilities, batch laid out as
            ``[source..., target...]``. Only the target half is used.
        """
        bsz, heads, spatial, tokens = attn_4d.shape
        half = bsz // 2
        if half == 0:
            return  # No target half — single-prompt run, nothing to record

        # Slice target half, average across heads, select the configured
        # target tokens, average over selected tokens. Result: per-patch
        # attention to "the new content" tokens.
        target = attn_4d[half:].detach()  # [half, heads, spatial, tokens]
        target = target.mean(dim=1)        # [half, spatial, tokens]
        target_at_tokens = target[
            :, :, [i for i in self.target_token_indices if i < tokens]
        ]
        if target_at_tokens.shape[-1] == 0:
            return  # All target_token_indices were out of range
        per_patch = target_at_tokens.mean(dim=-1)  # [half, spatial]

        # Resample to the canonical resolution.
        side = int(round(math.sqrt(spatial)))
        if side * side != spatial:
            # Non-square attention map; skip.
            return
        per_patch_2d = per_patch.reshape(half, 1, side, side)
        per_patch_2d = F.interpolate(
            per_patch_2d.float(),
            size=(self.base_resolution, self.base_resolution),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)  # [half, base_resolution, base_resolution]

        # Average across the target batch (usually half=1, but support
        # CFG batch=4 → half=2).
        per_patch_2d = per_patch_2d.mean(dim=0)  # [base_res, base_res]

        # Accumulate
        if self._accumulator_sum is None:
            self._accumulator_sum = per_patch_2d
            self._accumulator_count = 1
        else:
            self._accumulator_sum = self._accumulator_sum + per_patch_2d
            self._accumulator_count += 1

    # ------------------------------------------------------------------
    # Outputs to both controllers
    # ------------------------------------------------------------------

    def get_mask(self, spatial: int) -> torch.Tensor | None:
        """Return the current mask resampled to ``sqrt(spatial)`` x
        ``sqrt(spatial)``, flattened to length ``spatial``.

        Returns None at denoising step 0 (no mask available yet) or
        when no cross-attention has been recorded.

        Parameters
        ----------
        spatial : int
            The spatial dimension of the consumer's attention tensor
            (i.e., the number of patches at the consuming layer's
            resolution).
        """
        if self._current_mask is None:
            return None

        side = int(round(math.sqrt(spatial)))
        if side * side != spatial:
            return None

        if (
            self._current_mask.shape[-1] == side
            and self._current_mask.shape[-2] == side
        ):
            return self._current_mask.reshape(spatial)

        mask_2d = F.interpolate(
            self._current_mask.unsqueeze(0).unsqueeze(0).float(),
            size=(side, side),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)  # [side, side]
        # Re-binarize after interpolation if dilate_iters > 0; otherwise
        # use the soft mask directly.
        return mask_2d.reshape(spatial).to(self._current_mask.dtype)

    # ------------------------------------------------------------------
    # Per-step finalization (called by both controllers; idempotent)
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Finalize the current step's mask from accumulated cross-attention,
        store it as the mask to use for the NEXT step, and reset the
        accumulator.

        Idempotent: callable multiple times within a single denoising
        step (only the first call finalizes). Reserved for the case
        where multiple controllers all call this; in the P2P-only
        configuration, ``CrossAttentionController.step`` is the only
        caller.
        """
        if self._cur_step == self._finalized_step_id:
            return  # Already finalized for this step
        self._finalized_step_id = self._cur_step

        if self._accumulator_sum is not None and self._accumulator_count > 0:
            avg = self._accumulator_sum / self._accumulator_count

            # Normalize by max activation
            max_val = avg.max() + 1e-8
            normalized = avg / max_val

            # Threshold
            binary = (normalized > self.threshold).float()

            # Optional dilation via max-pool
            if self.dilate_iters > 0:
                m = binary.unsqueeze(0).unsqueeze(0)
                for _ in range(self.dilate_iters):
                    m = F.max_pool2d(m, kernel_size=3, stride=1, padding=1)
                binary = m.squeeze(0).squeeze(0)

            self._current_mask = binary

        # Reset accumulator for the NEXT step.
        self._accumulator_sum = None
        self._accumulator_count = 0
        self._cur_step += 1

    def reset(self) -> None:
        """Clear all state. Call between editing runs."""
        self._current_mask = None
        self._accumulator_sum = None
        self._accumulator_count = 0
        self._finalized_step_id = -1
        self._cur_step = 0
