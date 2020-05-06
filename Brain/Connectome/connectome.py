from .components import BrainPart


class Connectome:
    """
    Represent the graph of connections between areas and stimuli of the brain.
    This is a generic abstract class which offer a good infrastructure for building new models of connectome.
    You should implement some of the parts which are left for private case. For example when and how the connectome
    should be initialized, how the connections are represented.
	
	Attributes:
		brain_parts - List of all brain parts of the connectome.
		connections - Should maintain the graph of connection between all brain parts in the connectome.
		winners - List of winners for every brain part.
    """

    def __init__(self, parts=None, connections=None, winners=None):
        self.brain_parts: List[BrainPart] = []
        self.connections: Dict[Tuple[BrainPart, BrainPart], Connection] = {}
		self.winners: Dict[BrainPart, List[int]] = {}

		if brain_parts is not None:
			self.brain_parts = brain_parts

		if connections is not None:
			self.connections = connections

		if winners is not None:
			self.winners = winners
		
    def add_brain_part(self, brain_part:BrainPart):
        self.brain_parts.append(brain_part)

   
	def next_round(self, active_connections: Dict[BrainPart, BrainPart]):
		"""
		Runs one iteration of the connectome, update plasticity and winners in the connectome.
		This will be valid only to a subset of the available connections of the brain.
		:param active_connections: Mapping which maps each source brain part to a destination brain part.
		"""
		

    def __repr__(self):
        return f'{self.__class__.__name__} with {len(self.areas)} areas, and {len(self.stimuli)} stimuli'
