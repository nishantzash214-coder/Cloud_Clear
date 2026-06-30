"""Compatibility wrapper to expose `unetplusplus.CloudDetector` at
`src.models.cloud_detection.unetplusplus` for IDEs and imports."""

from unetplusplus import CloudDetector

__all__ = ["CloudDetector"]
