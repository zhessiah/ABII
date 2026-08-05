REGISTRY = {}

from .basic_controller import BasicMAC
from .non_shared_controller import NonSharedMAC
from .maddpg_controller import MADDPGMAC
from .centralized_controller import CentralizedMAC
from .non_shared_centralized_controller import NonSharedCentralizedMAC
from .adv_controller import ADVMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["centralized_mac"] = CentralizedMAC
REGISTRY["non_shared_mac"] = NonSharedMAC
REGISTRY["maddpg_mac"] = MADDPGMAC
REGISTRY["ns_centralized_mac"] = NonSharedCentralizedMAC
# 引入信念机制后的controller
REGISTRY["adv_mac"] = ADVMAC