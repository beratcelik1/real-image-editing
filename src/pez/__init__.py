"""PEZ — Hard Prompts Made Easy (Wen et al. 2023), adapted for our pipeline.

The submodule at ``external/pez/`` is the published reference. We
adapt the algorithm rather than vendor it because (a) their
``optim_utils.optimize_prompt_loop`` bakes in a CLIP-image-similarity
loss that we want to swap for SDS-CFG, and (b) their ``nn_project``
depends on ``sentence_transformers`` (a heavy dependency for one
utility function). See docs/pez_conditional/DESIGN_CHOICES.md.

Public exports:
- ``pez_search``: core optimization loop with a pluggable loss.
- ``pez_invert_source``: alternating-R=2 source-image inversion.
- ``pez_invert_with_instruction``: PEZ-2 (instruction-conditioned).
- ``clip_similarity_loss``, ``sds_cfg_loss``: the two loss families.
"""

from src.pez.search import pez_search
from src.pez.losses import clip_similarity_loss, sds_cfg_loss
from src.pez.source_inversion import pez_invert_source
from src.pez.instruction_conditioned import pez_invert_with_instruction

__all__ = [
    "pez_search",
    "clip_similarity_loss",
    "sds_cfg_loss",
    "pez_invert_source",
    "pez_invert_with_instruction",
]
