from itertools import product, chain
from numpy.core._multiarray_umath import ndarray
from typing import Dict, List, Tuple

from wrapt import ObjectProxy

from ..components import Area, BrainPart, Stimulus
from .connectome import Connectome


class NonLazyRandomConnectome(Connectome):
	"""
	Implementation of Non lazy random based connectome, based on the generic connectome.
	The object representing the connection in here is ndarray from numpy

	Attributes:
		(All the attributes of Connectome
		p: The probability for each edge of the connectome to exist
		initialize: Whether or not to fill the connectome of the brain in each place the connections are missing. If
		this is a subconnectome the initialize flag should be False
	"""

	def __init__(self, p: float, areas=None, stimuli=None, connections=None, initialize=True):
		"""

		:param p: The attribute p for the probability of an edge to exits
		:param areas: list of areas
		:param stimuli: list of stimuli
		:param connections: Optional argument which gives active connections to the connectome
		:param initialize: Whether or not to initialize the connectome of the brain.
		"""
		super(NonLazyRandomConnectome, self).__init__(areas, stimuli)
		self.p = p
		self.connections: Dict[Tuple[BrainPart, Area], ObjectProxy[ndarray]] = connections

		if connections is None:
			self.connections = dict([(conn, []) for conn in product(self.stimuli + self.areas, self.areas)])

		if initialize:
			self._initialize_parts(areas + stimuli)

	def add_area(self, area: Area):
		super(self).add_area(area)
		self._initialize_parts([area])

	def add_stimulus(self, stimulus: Stimulus):
		super(self).add_stimulus(stimulus)
		self._initialize_parts([stimulus])

	def _initialize_parts(self, parts: List[BrainPart]):
		"""
		Initialize all the connections to and from the given brain parts.
		:param parts: List of stimuli and areas to initialize
		:return:
		"""
		for part in parts:
			for other in self.areas + self.stimuli:
				self._initialize_connection(part, other)
				if isinstance(part, Area) and part != other:
					self._initialize_connection(other, part)

	def _initialize_connection(self, part: BrainPart, area: Area):
		"""
		Initalize the connection from brain part to an area
		:param part: Stimulus or Area which the connection should come from
		:param area: Area which the connection go to
		:return: 
		"""""
		pass  # TODO

	def get_connection(self, source: BrainPart, sink: Area) -> ndarray:
		return self.connections[(source, sink)][0]

	def set_connection(self, source: BrainPart, sink: Area, conn: ndarray):
		self.connections[(source, sink)][0] = conn

	def get_subconnectome(self, connections: Dict[BrainPart, List[Area]]) -> Connectome:
		areas = set([part for part in connections if isinstance(part, Area)] + list(chain(*connections.values())))
		stimuli = [part for part in connections if isinstance(part, Stimulus)]
		edges = [(part, area) for part in connections for area in connections[part]]
		neural_subnet = [(edge, self.connections[edge]) for edge in edges]
		return NonLazyRandomConnectome(self.p, areas=list(areas), stimuli=stimuli, connections=neural_subnet,
		                               initialize=False)

	def get_connections_to_area(self, area: Area) -> List[BrainPart]:
		connections = []
		for conn in self.connections:
			if conn[1] == area:
				connections.append(conn[0])
		return connections
