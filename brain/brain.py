from typing import Dict, Set

from .brain_recipe import BrainRecipe
from .components import BrainPart
from .connectome import Connectome
from brain.components import Area, UniquelyIdentifiable


class Brain(UniquelyIdentifiable):
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

	def __init__(self, connectome: Connectome, recipe: BrainRecipe = None):
		super(Brain, self).__init__()
		self.connectome: Connectome = connectome
		self.active_connectome: Dict[BrainPart, Set[BrainPart]] = {}
		self.recipe = recipe or None

	def next_round(self):
		return self.connectome.next_round(self.active_connectome)

	def add_brain_part(self, brain_part: BrainPart):
		self.recipe.append(brain_part)
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

	def get_winners(self, area: Area):
		pass

	def get_support(self, area: Area):
		pass

	def __enter__(self):
		for area in self.recipe.areas:
			area.bind(brain=self)

		for assembly in self.recipe.assemblies:
			assembly.bind(brain=self)

	def __exit__(self, exc_type, exc_val, exc_tb):
		for area in self.recipe.areas:
			area.bind('brain')

		for assembly in self.recipe.assemblies:
			assembly.bind('brain')


def bake(recipe: BrainRecipe, connectome_cls):
	return Brain(connectome_cls(), recipe=recipe)
