"""Top-level package for MSM_a7_nachrs."""

__author__ = """Yuxuan Zhuang"""
__email__ = "yuxuan.zhuang@dbb.su.se"
__version__ = "0.1.0"

from .vampnet import (
    VAMPNETInitializer,
    VAMPNETInitializer_Multimer,
    VAMPNet_Multimer,
    VAMPNet_Multimer_AUG,
)

from .lobe import (
    MultimerNet,
    MultimerNet_200,
    MultimerNet_400,

)

from .vampnet_rev import VAMPNet_Multimer_REV

from .vampnet_sym import VAMPNet_Multimer_SYM, VAMPNet_Multimer_SYM_REV

# from .vampnet_rev import (
#            VAMPNet_Multimer_Rev_Model,
#            VAMPNet_Multimer_Rev,
#            )

# from .srv import (
#            VAMPNet_Multimer_Sym,
#            VAMPNet_Multimer_Sym_NOSYM
#            )
