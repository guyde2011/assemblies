from numpy.core._multiarray_umath import ndarray
from typing import Dict, List, Tuple

from wrapt import ObjectProxy

from ..components import Area, BrainPart, Stimulus
from .connectome import Connectome


class LazyRandomConnectome(Connectome):
	def __init__(self, p: float, areas=None, stimuli=None):
		super(LazyRandomConnectome, self).__init__(areas, stimuli)
		self.p = p
		self.connections: Dict[Tuple[BrainPart, Area], ObjectProxy(ndarray)] = {}

	def get_connection(self, source: BrainPart, sink: Area) -> ObjectProxy(ndarray):
		pass

	def set_connection(self, source: BrainPart, sink: Area, conn: ObjectProxy(ndarray)):
		pass

	def get_subconnectome(self, connections: Dict[BrainPart, Area]) -> Connectome:
		pass

	def get_connections_to_area(self, area: Area) -> List[Area]:
		pass
