"""Top-level package for sym_msm.decomposition"""

__author__ = """Yuxuan Zhuang"""
__email__ = "yuxuan.zhuang@dbb.su.se"
__version__ = "0.1.0"

from .tica_initializer import (
    TICAInitializer,
    SymTICAInitializer,
)

from .sym_tica import (
    SymVAMP,
    SymTICA,
)
