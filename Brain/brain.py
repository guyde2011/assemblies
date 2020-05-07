from typing import Dict, List

from .Connectome import BrainPart, Connectome


# Library Ext team:
class Assembly:
	pass


class Brain:
	T = 10
	"""
	Represents a simulated brain, with it's connectome which holds the areas, stimuli, and all the synapse weights.
	The brain updates by selecting a subgraph of stimuli and areas, and activating only those connections.
	The brain object works with a general connectome, which export an unified api for how the connections between the
	parts of the brain should be used. In case of need, one should extend the Connectome API as he would like to make
	the implementation of the brain easier/better. Note that the brain implementation shouldn't depends on the
	underlying implementation of the connectome.
	
	Attributes:
		connectome: The full connectome of the brain, hold all the connections between the brain parts.
		active_connectome: The current active subconnectome of the brain. Gives a nice way of supporting inhibit, disinhibit.   
	
	"""

	def __init__(self, connectome, active_connectome=None):
		self.connectome: Connectome = connectome
		self.active_connectome: Dict[BrainPart, Set[BrainPart]] = {}
		if active_connectome is not None:
			self.active_connectome = active_connectome

	def next_round(self):
		return self.connectome.next_round(self.active_connectome)

	def add_brain_part(self, brain_part: BrainPart):
		return self.connectome.add_brain_part(brain_part)

	next_round.__doc__ = Connectome.next_round.__doc__
	add_brain_part.__doc__ = Connectome.add_brain_part.__doc__

	def inhibit(self, source: BrainPart, dest: BrainPart = None):
		"""
		Inhibit connection between two brain parts (i.e. activate it).
		If dest is None then all connections from the source are inhibited.
		:param source: The source brain part of the connection.
		:param dest: The destination brain part of the connection.
		"""
		if dest is not None:
			self.active_connectome[source].discard(dest)
			return
		for sink in self.connectome.brain_parts:
			self.inhibit(source, sink)

	def disinhibit(self, source: BrainPart, dest: BrainPart = None):
		"""
		Disinhibit connection between two brain parts (i.e. deactivate it).
		If dest is None then all connections from the source are disinhibited.
		:param source: The source brain part of the connection.
		:param dest: The destination brain part of the connection.
		"""
		if dest is not None:
			self.active_connectome[source].add(dest)
			return
		for sink in self.connectome.brain_parts:
			self.disinhibit(source, sink)

	# Library Ext for research:
	def project(self, x: Assembly, brain_part: BrainPart) -> Assembly:
		pass

	def reciprocal_project(self, x: Assembly, brain_part: BrainPart) -> Assembly:
		pass

	def association(self, x: Assembly, y: Assembly):
		pass

	def merge(self, x: Assembly, y: Assembly, area: BrainPart) -> Assembly:
		pass
