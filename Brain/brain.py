import math
from typing import Dict, List

from .components import Area, BrainPart, Stimulus
from .Connectome import Connectome


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
		active_connectome: The current active subconnectome  of the brain, note that this connectome should be a valid
		subconnectome of the original connectome, so the changes of on the subconnectome will reflect in the original 
		connectome. 
		# Optional: make the entire brain interface to depend on the actual brain parts and connections like
		# in the original project_into.
		winners: The last fired neurons of every area.
		#To be continued by library ext. for research
	
	# Advantages in using the Connectome abstraction: (TL;DR A lot of Gittik's bullshit)
	# 1. Progressive disclose: When reading the previous implementation one can easily be confused about how
	#	 get along. By abstracting the connectome from the way it's built, we can easily understand the abstract api for 
	#    connectome, understand it's methods, which makes the understanding of the brain class much easier.
	#    If later we would like to know how the connectome is built, we can simply pick a subclass and read it's 
	#    implementation.
	# 2. Future Comparability: If later we would like to extend the api or change the underlying implementation of the 
	#    connectome, we can do so, without touching the code of the brain. Just extend the connectome api or inherit it
	#    with a different subclass.
	# 3. Easy for benchmarking and testing: By giving few models of the brain we can hold every single implementation of
	#    them simultaneously, run them in parallel. If we want to the test correctness, now we can test for Brain and
	#    connectome separately.
	"""

	def __init__(self, connectome: Connectome):
		self.connectome: Connectome = connectome
		self.active_connectome: Connectome = connectome
		self.winners: Dict[Area, List[int]] = {}

	def add_area(self, area: Area):
		self.connectome.add_area(area)

	def add_stimulus(self, stimulus: Stimulus):
		self.connectome.add_stimulus(stimulus)

	# Performance:
	def next_round(self, subconnetome: Connectome = None):
		"""
		Calculate the next set of winners in each part of the given subconnectome.
		(Equivalent to previous project_into/project)

		:param subconnetome: subgraph of the given connectome. If subconnectome is None use the active subconnectome
		of the brain
		:return:
		"""
		if subconnetome is None:
			subconnetome = self.active_connectome
		for _ in subconnetome.areas:  # replace _ with area
			pass  # TODO: calculate the winners of this area of the brain

	# Library Ext for research:
	def project(self, x: Assembly, area: Area) -> Assembly:
		pass

	def reciprocal_project(self, x: Assembly, area: Area) -> Assembly:
		pass

	def association(self, x: Assembly, y: Assembly):
		pass

	def merge(self, x: Assembly, y: Assembly, area: Area) -> Assembly:
		pass
