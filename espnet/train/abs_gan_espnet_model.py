# espnet/train/abs_gan_espnet_model.py
import warnings
warnings.warn(
    "Importing AbsESPnetModel from espnet.train is deprecated. "
    "Use espnet.models.abs_espnet_model instead.",
    DeprecationWarning, stacklevel=2,
)
from espnet.models.abs_gan_espnet_model import AbsESPnetModel