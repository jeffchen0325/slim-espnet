"""SPGISpeech dataset module."""

from egs.spgispeech.asr.dataset.builder import SPGISpeechBuilder as DatasetBuilder
from egs.spgispeech.asr.dataset.dataset import SPGISpeechDataset as Dataset
from egs.spgispeech.asr.dataset.dataset import gather_training_text

__all__ = ["Dataset", "DatasetBuilder", "gather_training_text"]
