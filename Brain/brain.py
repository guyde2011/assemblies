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
		self.active_connectome: Dict[BrainPart, List[BrainPart]] = {}
		if active_connectome is not None:
			self.active_connectome = active_connectome

	def next_round(self):
		return self.connectome.next_round(self.active_connectome)

	def add_brain_part(self, brain_part: BrainPart):
		return self.connectome.add_brain_part(brain_part)

	next_round.__doc__ = Connectome.next_round.__doc__
	add_brain_part.__doc__ = Connectome.add_brain_part.__doc__

	def inhibit_connection(self, source: BrainPart, dest: BrainPart):
		"""
		Inhibit a single connection i.e. enabling it's firing
		:param source: The source brain part of the connection.
		:param dest: The destination brain part of the connection.
		"""

	def inhibit_brain_part(self, brain_part: BrainPart):
		"""
		Inhibit the brain part, which enables it to fire to other connections in the brain.
		:param brain_part: BrainPart which is part of the connectome.
		"""
		for sink in self.connectome.brain_parts:
			self.inhibit_connection(brain_part, sink)

	def disinhibit_connection(self, source: BrainPart, dest: BrainPart):
		"""
		Disinhibit a single connection i.e. disables it from firing
		:param source: The source brain part of the connection.
		:param dest: The destination brain part of the connection.
		"""

	def disinhibit_brain_part(self, brain_part: BrainPart):
		"""
		Fully disinhibit the brain_part from firing onto every other brain part.
		The part can still be fired from other brain parts.
		:param brain_part: BrainPart which is part of the connectome.
		"""
		for sink in self.connectome.brain_parts:
			self.disinhibit_connection(brain_part, sink)

	# Library Ext for research:
	def project(self, x: Assembly, brain_part: BrainPart) -> Assembly:
		pass

	def reciprocal_project(self, x: Assembly, brain_part: BrainPart) -> Assembly:
		pass

	def association(self, x: Assembly, y: Assembly):
		pass

	def merge(self, x: Assembly, y: Assembly, area: BrainPart) -> Assembly:
		pass
