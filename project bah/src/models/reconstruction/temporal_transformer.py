"""Compatibility wrapper for `temporal_transformer`.

Re-exports `TemporalTransformer` for imports under
`src.models.reconstruction.temporal_transformer`.
"""

from temporal_transformer import TemporalTransformer

__all__ = ["TemporalTransformer"]
