"""LibriSpeech 100h dataset module."""

from egs.librispeech_100.asr.dataset.builder import (
    LibriSpeech100Builder as DatasetBuilder,
)
from egs.librispeech_100.asr.dataset.dataset import LibriSpeech100Dataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
