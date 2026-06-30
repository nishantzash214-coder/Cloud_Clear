"""Compatibility wrapper for cross-attention fusion.

Re-exports core classes used by the reconstruction pipeline.
"""

from cross_attention_fusion import OpticalQueryEncoder, CrossAttentionFusionLayer, build_fusion_layer

__all__ = ["OpticalQueryEncoder", "CrossAttentionFusionLayer", "build_fusion_layer"]
