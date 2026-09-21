# espnet3/train/abs_espnet_model.py
import warnings
warnings.warn(
    "Importing AbsESPnetModel from espnet3.train is deprecated. "
    "Use espnet3.models.abs_espnet_model instead.",
    DeprecationWarning, stacklevel=2,
)
from espnet.models.abs_espnet_model import AbsESPnetModel
