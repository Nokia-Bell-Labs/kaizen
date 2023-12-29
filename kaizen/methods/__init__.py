from kaizen.methods.barlow_twins import BarlowTwins
from kaizen.methods.base import BaseModel
from kaizen.methods.byol import BYOL
from kaizen.methods.deepclusterv2 import DeepClusterV2
from kaizen.methods.dino import DINO
from kaizen.methods.linear import LinearModel
from kaizen.methods.mocov2plus import MoCoV2Plus
from kaizen.methods.nnclr import NNCLR
from kaizen.methods.ressl import ReSSL
from kaizen.methods.simclr import SimCLR
from kaizen.methods.simsiam import SimSiam
from kaizen.methods.swav import SwAV
from kaizen.methods.vicreg import VICReg
from kaizen.methods.wmse import WMSE
from kaizen.methods.full_model import FullModel

METHODS = {
    # base classes
    "base": BaseModel,
    "linear": LinearModel,
    "full_model": FullModel,
    # methods
    "barlow_twins": BarlowTwins,
    "byol": BYOL,
    "deepclusterv2": DeepClusterV2,
    "dino": DINO,
    "mocov2plus": MoCoV2Plus,
    "nnclr": NNCLR,
    "ressl": ReSSL,
    "simclr": SimCLR,
    "simsiam": SimSiam,
    "swav": SwAV,
    "vicreg": VICReg,
    "wmse": WMSE,
}
__all__ = [
    "BarlowTwins",
    "BYOL",
    "BaseModel",
    "DeepClusterV2",
    "DINO",
    "LinearModel",
    "FullModel",
    "MoCoV2Plus",
    "NNCLR",
    "ReSSL",
    "SimCLR",
    "SimSiam",
    "SwAV",
    "VICReg",
    "WMSE",
]

try:
    from kaizen.methods import dali  # noqa: F401
except ImportError:
    pass
else:
    __all__.append("dali")
