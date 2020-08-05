from __future__ import annotations

from collections import defaultdict
from functools import cached_property
from typing import Dict, Set, TYPE_CHECKING, List, Optional, Union

from .components import BrainPart, Stimulus
from brain.connectome.abstract_connectome import AbstractConnectome
from brain.components import Area, UniquelyIdentifiable
# TODO: imports should happen in any case
if TYPE_CHECKING:
	from .brain_recipe import BrainRecipe
	from assemblies import Assembly


class Brain(UniquelyIdentifiable):
	# TODO: is T used?
	T = 10
	"""
	Represents a simulated brain, with it's connectome which holds the areas, stimuli, and all the synapse weights.
	The brain updates by selecting a subgraph of stimuli and areas, and activating only those connections.
	The brain object works with a general connectome, which export an unified api for how the connections between the
	parts of the brain should be used. In case of need, one should extend the connectome API as he would like to make
	the implementation of the brain easier/better. Note that the brain implementation shouldn't depends on the
	underlying implementation of the connectome.
	
	Attributes:
		connectome: The full connectome of the brain, hold all the connections between the brain parts.
		active_connectome: The current active subconnectome of the brain. Gives a nice way of supporting inhibit, disinhibit.   
	
	"""

	def __init__(self, connectome: AbstractConnectome, recipe: BrainRecipe = None, repeat: int = 1):
		# TODO: document __init__ parameters
		super(Brain, self).__init__()
		self.repeat = repeat
		self.recipe = recipe or BrainRecipe()
		self.connectome: AbstractConnectome = connectome
		self.active_connectome: Dict[BrainPart, Set[BrainPart]] = defaultdict(lambda: set())
		self.ctx_stack: List[Dict[Union[BrainPart, Assembly], Optional[Brain]]] = []

		for area in self.recipe.areas:
			self.add_area(area)

		for stimulus in self.recipe.stimuli:
			self.add_stimulus(stimulus)

	# TODO: make uniform use of type hints (if exists in some functions, add to all functions)
	# TODO 2: `subconnectome` is never passed as None, is this option relevant?
	# TODO 6: this function is confusing: it depends on `replace` state, behaves differently if `subconnectome` is None or not,
	# TODO 6: performs a merge operation between `active_connectome` and `subconnectome`, and returns an undefined value.
	# TODO 6: please make it clearer and simplify the logic
	def next_round(self, subconnectome=None, replace=False, iterations=1):
		# TODO 3: make next statement clearer
		if replace or subconnectome is None:
			_active_connectome = subconnectome or self.active_connectome
		else:
			# TODO 4: the following rows should use dictionary merge logic
			_active_connectome = self.active_connectome.copy()
			for source, destinations in subconnectome.items():
				for dest in destinations:
					_active_connectome[source].add(dest)

		for _ in range(iterations - 1):
			self.connectome.project(_active_connectome)
		# TODO 5: `project` in `Connectome` class has no `return` - what is expected to be returned here?
		return self.connectome.project(_active_connectome)

	def add_area(self, area: Area):
		self.recipe.append(area)
		self.connectome.add_area(area)
		self.enable(area, area)

	def add_stimulus(self, stimulus: Stimulus):
		self.recipe.append(stimulus)
		return self.connectome.add_stimulus(stimulus)

	def enable(self, source: BrainPart, dest: BrainPart = None):
		"""
		# TODO: "inhibit" means to disable connections
		Inhibit connection between two brain parts (i.e. activate it).
		If dest is None then all connections from the source are inhibited.
		:param source: The source brain part of the connection.
		:param dest: The destination brain part of the connection.
		"""
		if dest is not None:
			self.active_connectome[source].add(dest)
			return
		for sink in self.connectome.areas + self.connectome.stimuli:
			self.enable(source, sink)

	def disable(self, source: BrainPart, dest: BrainPart = None):
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
			self.disable(source, sink)

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


# TODO: document
# TODO 2: typehint `connectome_cls`
# TODO 3: is it crucial to get `connectome_cls` or can we get a connectome object?
# TODO 4: make names clearer: train_repeat -> recipe_repeat, effective_repeat -> something clearer?
# TODO 5: should this be a method of `BrainRecipe`?
def bake(recipe: BrainRecipe, p: float, connectome_cls, train_repeat: int, effective_repeat: int = 1):
	brain = Brain(connectome_cls(p), recipe=recipe, repeat=train_repeat)
	recipe.initialize(brain)
	brain.repeat = effective_repeat
	return brain
