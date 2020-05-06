from numpy.core._multiarray_umath import ndarray
from typing import Dict, List, Tuple

from wrapt import ObjectProxy

from ..components import Area, BrainPart, Stimulus
from .connectome import Connectome


class LazyConnectome(Connectome):
	def __init__(self, p: float, areas=None, stimuli=None):
		super(LazyConnectome, self).__init__(p, areas, stimuli)


	def subconnectome(self, connections: Dict[BrainPart, Area]) -> Connectome:
		pass

	def area_connections(self, area: Area) -> List[Area]:
		pass
