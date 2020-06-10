from __future__ import annotations

from collections import defaultdict
from functools import cached_property
from typing import Dict, Set, TYPE_CHECKING, List, Optional, Union

from .components import BrainPart, Stimulus
from .connectome import Connectome
from brain.components import Area, UniquelyIdentifiable
if TYPE_CHECKING:
	from .brain_recipe import BrainRecipe
	from assemblies import Assembly


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

	def __init__(self, connectome: Connectome, recipe: BrainRecipe = None, t: int = 1):
		super(Brain, self).__init__()
		self.t = t
		self.recipe = recipe or BrainRecipe()
		self.connectome: Connectome = connectome
		self.active_connectome: Dict[BrainPart, Set[BrainPart]] = defaultdict(lambda: set())
		self.ctx_stack: List[Dict[Union[BrainPart, Assembly], Optional[Brain]]] = []

	def next_round(self):
		return self.connectome.project(self.active_connectome)

	def add_area(self, area: Area):
		self.recipe.append(area)
		return self.connectome.add_area(area)

	def add_stimulus(self, stimulus: Stimulus):
		self.recipe.append(stimulus)
		return self.connectome.add_stimulus(stimulus)

	def inhibit(self, source: BrainPart, dest: BrainPart = None):
		"""
		Inhibit connection between two brain parts (i.e. activate it).
		If dest is None then all connections from the source are inhibited.
		:param source: The source brain part of the connection.
		:param dest: The destination brain part of the connection.
		"""
		if dest is not None:
			self.active_connectome[source].add(dest)
			return
		for sink in self.connectome.areas + self.connectome.stimuli:
			self.inhibit(source, sink)

	def disinhibit(self, source: BrainPart, dest: BrainPart = None):
		"""
		Disinhibit connection between two brain parts (i.e. deactivate it).
		If dest is None then all connections from the source are disinhibited.
		:param source: The source brain part of the connection.
		:param dest: The destination brain part of the connection.
		"""
		if dest is not None:
			self.active_connectome[source].discard(dest)
			return
		for sink in self.connectome.areas:
			self.disinhibit(source, sink)

	@cached_property
	def winners(self):
		return self.connectome.winners

	@cached_property
	def support(self):
		# TODO: Implement
		return None

	def __enter__(self):
		current_ctx_stack: Dict[Union[BrainPart, Assembly], Optional[Brain]] = {}

		for area in self.recipe.areas:
			if 'brain' in area.bound_params:
				current_ctx_stack[area] = area.bound_params['brain']
			area.bind(brain=self)

		for assembly in self.recipe.assemblies:
			if 'brain' in assembly.bound_params:
				current_ctx_stack[assembly] = assembly.bound_params['brain']
			assembly.bind(brain=self)

		self.ctx_stack.append(current_ctx_stack)
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		current_ctx_stack: Dict[Union[BrainPart, Assembly], Optional[Brain]] = self.ctx_stack.pop()

		for area in self.recipe.areas:
			area.unbind('brain')
			if area in current_ctx_stack:
				area.bind(brain=current_ctx_stack[area])

		for assembly in self.recipe.assemblies:
			assembly.unbind('brain')
			if assembly in current_ctx_stack:
				assembly.bind(brain=current_ctx_stack[assembly])


def bake(recipe: BrainRecipe, p: float, connectome_cls, t: int = 1):
	brain = Brain(connectome_cls(p, areas=recipe.areas, stimuli=recipe.stimuli), recipe=recipe, t=t)
	recipe.initialize(brain)
	return brain
