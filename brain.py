""" Configurable brain assembly model for simulations and research.
Author: Daniel Mitropolsky, 2018

This module contains classes to represent different elements of a brain simulation:
	- Area - Represents an individual area of the brain, with the relevant parameters.
	- Connectomes are the connections between neurons. They have weights, which are initialized randomly but
		due to plasticity they can updated every time some neuron fires. These weights are represented by numpy arrays.
		The ones that are not random, because they were influenced by previous projections, are referred to as the 'support'.
	- Winners in a given 'round' are the specific neurons that fired in that round.
		In any specific area, these will be the 'k' neurons with the highest value flown into them.
		These are also the only neurons whose connectome weights get updated. #TODO: Accurate?
	- Stimulus - Represents a random stimulus that can be applied to any part of the brain.
		When a stimulus is created it is initialized randomly, but when applied multiple times this will change.
		This is equivalent to k neurons from an unknown part of the brain firing and their (initially, random)
		connectomes decide how this stimulus affects a given area of the brain.
	- Brain - A class representing a simulated brain, with it's different areas, stimulus, and all the connectome weights.
		A brain is initialized as a random graph, and it is maintained in a 'sparse' representation,
		meaning that all neurons that have their original, random connectome weights (0 or 1) are not saved explicitly,
		rather handled as a group for all calculations. #TODO: Does this have any limitations?
	- Assembly - TODO define and express in code
"""
from typing import List, Mapping, Tuple
import numpy as np
import heapq
from collections import defaultdict
from scipy.stats import binom
from scipy.stats import truncnorm
import math
import random

# TODO Document all classes and methods
# TODO Add type hints for everything
# TODO Improve the code readability


class Stimulus:
	""" Represents a random stimulus that can be applied to any part of the brain.

	Being random, the only thing the programmer needs to provide is
	the number k of neurons that fire to create this stimulus.

	Attributes:
		k: number of neurons that fire
	"""
	def __init__(self, k: int):
		self.k = k


class Area:
	"""Represents an individual area of the brain, with the relevant parameters.

	Since it is initialized randomly, all the programmer needs to provide is the number 'n' of neurons,
	number 'k' of winners in any given round (meaning the k neurons with heights values will fire),
	and the parameter 'beta' of plasticity controlling connectome weight updates.
	# TODO: Seems there are more betas, understand and explain them

	Attributes:
		n: number of neurons
		k: number of winners
		beta:
		stimulus_beta:
		area_beta:
		w:
		winners:
		_new_w:
		_new_winners:
		saved_winners:
		saved_w:
		num_first_winners:
	"""
	def __init__(self, name: str, n: int, k: int, beta: float = 0.05):
		self.name = name
		self.n = n
		self.k = k
		# Default beta
		self.beta = beta
		# Betas from stimuli into this area.
		self.stimulus_beta = {}
		# Betas form areas into this area.
		self.area_beta = {}
		# Size of the support, i.e. the number of connectomes with non-random values
		self.w = 0
		# List of winners currently (after previous action). Can be 
		# read by caller.
		self.winners = []
		# new winners computed DURING a projection, do not use outside of internal project function
		self._new_w = 0
		self._new_winners = []
		# list of lists of all winners in each round, only saved if user asks for it
		self.saved_winners = []
		# list of size of support in each round, only saved if user asks for it
		self.saved_w = []
		self.num_first_winners = -1

	def update_winners(self):
		""" TODO
		"""
		self.winners = self._new_winners
		self.w = self._new_w

	def update_stimulus_beta(self, name: str, new_beta: float):
		""" TODO
		"""
		self.stimulus_beta[name] = new_beta

	def update_area_beta(self, name: str, new_beta: float):
		""" TODO
		"""
		self.area_beta[name] = new_beta


class Brain:
	"""Represents a simulated brain, with it's different areas, stimulus, and all the connectome weights.
	Being a randomly initialized brain, all the programmer needs to provide is the probability 'p' of a
	connectome existing between any two given neurons.

	Attributes:
		areas: A mapping from area names to Area objects representing them.
		stimuli: A mapping from stimulus names to Stimulus objects representing them.
		stimuli_connectomes: The connectome weights for each stimulus, saved sparsely only for non-trivial neurons,
		meaning neurons that had been winners in some projection (otherwise all connectomes are randomly 0 or 1).
		connectomes: The connectome weights for each area, saved sparsely only for non-trivial neurons.
		p: Probability of connectome (edge) existing between two neurons (vertices)
		save_size: Whether we should keep track of the size of the support TODO Remove this or implement in a different way? It's just ugly research code
		save_winners: TODO Remove this or implement in a different way? It's just ugly research code
	"""

	def __init__(self, p: float, save_size: bool = True, save_winners: bool = False):
		self.areas = {}
		self.stimuli = {}
		self.stimuli_connectomes = {}
		self.connectomes = {} 
		self.p = p
		self.save_size = save_size
		self.save_winners = save_winners

	def add_stimulus(self, name: str, k: int):
		""" Initialize a random stimulus with 'k' neurons firing.
		This stimulus can later be applied to different areas of the brain,
		also updating its outgoing connectomes in the process.

		:param name: Name used to refer to stimulus
		:param k: Number of neurons in the stimulus
		:return:
		"""
		self.stimuli[name] = Stimulus(k)
		new_connectomes = {}
		for key in self.areas:
			new_connectomes[key] = np.empty((0, 0))
			self.areas[key].stimulus_beta[name] = self.areas[key].beta
		self.stimuli_connectomes[name] = new_connectomes

	def add_area(self, name: str, n: int, k: int, beta: float):
		"""Add an area to this brain, randomly connected to all other areas and stimulus.

		The random connections are controlled by the global 'p' parameter of the brain,
		initializing each connectome to have a value of 0 or 1 with probability 'p'.

		:param name: Name of area
		:param n: Number of neurons in the new area
		:param k: Number of winners in the new area
		:param beta: plasticity parameter of connectomes coming INTO this area.
				The plasticity parameter of connectomes FROM this area INTO other areas are decided by the betas of those other areas.
		"""
		self.areas[name] = Area(name, n, k, beta)

		for stim_name, stim_connectomes in self.stimuli_connectomes.items():
			stim_connectomes[name] = np.empty(0)
			self.areas[name].stimulus_beta[stim_name] = beta

		new_connectomes = {}
		for key in self.areas:
			new_connectomes[key] = np.empty((0, 0))
			if key != name:
				self.connectomes[key][name] = np.empty((0, 0))
			self.areas[key].area_beta[name] = self.areas[key].beta
			self.areas[name].area_beta[key] = beta
		self.connectomes[name] = new_connectomes

	def update_plasticities(self, area_update_map: Mapping[str, List[Tuple[str, float]]] = {},
								stim_update_map: Mapping[str, List[Tuple[str, float]]] = {}):
		""" This is used to update the plasticity parameters of connectomes between areas and from stimuli into areas.
		TODO: What about within an area? Why is it not part of this function as well?

		:param area_update_map: dictionary containing, for each area, a list of incoming betas to be updated #TODO: Example
		:param stim_update_map: dictionary containing, for each area, a list of incoming betas to be updated #TODO: Example
		"""
		# area_update_map consists of area1: list[ (area2, new_beta) ]
		# represents new plasticity FROM area2 INTO area1
		for to_area, update_rules in list(area_update_map.items()):
			for (from_area, new_beta) in update_rules: 
				self.areas[to_area].area_beta[from_area] = new_beta

		# stim_update_map consists of area: list[ (stim, new_beta) ]f
		# represents new plasticity FROM stim INTO area
		for area, update_rules in list(stim_update_map.items()):
			for (stim, new_beta) in update_rules:
				self.areas[area].stimulus_beta[stim] = new_beta

	# TODO: Add default values like update_plasticities method?
	def project(self, stim_to_area: Mapping[str, List[str]],
					area_to_area: Mapping[str, List[str]],
					verbose=False):
		"""Projecting is what happens when a stimulus is applied to some area,
		and also when a resulting assembly formed in some area fires into a separate brain area, creating a secondary stimulus, etc.

		:param stim_to_area: Dictionary that matches to each stimuli applied a list of areas to project into.
			Example: {"stim1":["A"], "stim2":["C","A"]}
		:param area_to_area: Dictionary that matches for each area a list of areas to project into.
			Note that an area can also be projected into itself.
			Example: {"A":["A","B"],"C":["C","A"]}
		:param verbose: Print debug information #TODO: This should be done with logging package levels
		"""
		stim_in = defaultdict(lambda: [])
		area_in = defaultdict(lambda: [])
		# Validate stim_area, area_area well defined
		for stim, areas in stim_to_area.items():
			if stim not in self.stimuli:
				raise IndexError(stim + " not in brain.stimuli")
			for area in areas:
				if area not in self.areas:
					raise IndexError(area + " not in brain.areas")
				stim_in[area].append(stim)
		for from_area, to_areas in area_to_area.items():
			if from_area not in self.areas:
				raise IndexError(from_area + " not in brain.areas")
			for to_area in to_areas:
				if to_area not in self.areas:
					raise IndexError(to_area + " not in brain.areas")
				area_in[to_area].append(from_area)

		to_update = set().union(list(stim_in.keys()), list(area_in.keys()))

		for area in to_update:
			num_first_winners = self.project_into(self.areas[area], stim_in[area], area_in[area], verbose)
			self.areas[area].num_first_winners = num_first_winners
			if self.save_winners:
				self.areas[area].saved_winners.append(self.areas[area].new_winners)

		# once done everything, for each area in to_update: area.update_winners()
		for area in to_update:
			self.areas[area].update_winners()
			if self.save_size:
				self.areas[area].saved_w.append(self.areas[area].w)

	def project_into(self, area: Area, from_stimuli: List[str], from_areas: List[str], verbose: bool = False):
		"""Project multiple stimuli and area assemblies into area 'area' at the same time.

		:param area: The area projected into
		:param from_stimuli: The stimuli that we will be applying
		:param from_areas: List of separate areas whose assemblies we will project into this area
		:return: Returns the number of area neurons that were winners for the first time during this projection
		"""
		# projecting everything in from stim_in[area] and area_in[area]
		# calculate: inputs to self.connectomes[area] (previous winners)
		# calculate: potential new winners, Binomial(sum of in sizes, k-top)
		# k top of previous winners and potential new winners
		# if new winners > 0, redo connectome and intra_connectomes
		# have to wait to replace new_winners
		# TODO Add more documentation to this function which does most of the work
		# TODO (EDO): Handle case of projecting from an area without previous winners.
		print(("Projecting " + ",".join(from_stimuli) + " and " + ",".join(from_areas) + " into " + area.name))

		name = area.name
		prev_winner_inputs = [0.] * area.w
		for stim in from_stimuli:
			stim_inputs = self.stimuli_connectomes[stim][name]
			for i in range(area.w):
				prev_winner_inputs[i] += stim_inputs[i]
		for from_area in from_areas:
			connectome = self.connectomes[from_area][name]
			for w in self.areas[from_area].winners:
				for i in range(area.w):
					prev_winner_inputs[i] += connectome[w][i]

		if verbose:
			print("prev_winner_inputs: ")
			print(prev_winner_inputs)

		# simulate area.k potential new winners
		total_k = 0
		input_sizes = []
		num_inputs = 0
		for stim in from_stimuli:
			total_k += self.stimuli[stim].k
			input_sizes.append(self.stimuli[stim].k)
			num_inputs += 1
		for from_area in from_areas:
			# if self.areas[from_area].w < self.areas[from_area].k:
			#	raise ValueError("Area " + from_area + "does not have enough support.")
			effective_k = len(self.areas[from_area].winners)
			total_k += effective_k
			input_sizes.append(effective_k)
			num_inputs += 1

		if verbose:
			print("total_k = " + str(total_k) + " and input_sizes = " + str(input_sizes))

		effective_n = area.n - area.w
		# Threshold for inputs that are above (n-k)/n percentile.
		# self.p can be changed to have a custom connectivity into this brain area.
		alpha = binom.ppf((float(effective_n-area.k)/effective_n), total_k, self.p)
		if verbose:
			print(("Alpha = " + str(alpha)))
		# use normal approximation, between alpha and total_k, round to integer
		# create k potential_new_winners
		std = math.sqrt(total_k * self.p * (1.0-self.p))
		mu = total_k * self.p
		a = float(alpha - mu) / std
		b = float(total_k - mu) / std
		potential_new_winners = truncnorm.rvs(a, b, scale=std, size=area.k)
		for i in range(area.k):
			potential_new_winners[i] += mu
			potential_new_winners[i] = round(potential_new_winners[i])
		potential_new_winners = potential_new_winners.tolist()

		if verbose:
			print("potential_new_winners: ")
			print(potential_new_winners)

		# take max among prev_winner_inputs, potential_new_winners
		# get num_first_winners (think something small)
		# can generate area.new_winners, note the new indices
		both = prev_winner_inputs + potential_new_winners
		new_winner_indices = heapq.nlargest(area.k, list(range(len(both))), both.__getitem__)
		num_first_winners = 0
		first_winner_inputs = []
		for i in range(area.k):
			if new_winner_indices[i] >= area.w:
				first_winner_inputs.append(potential_new_winners[new_winner_indices[i] - area.w])
				new_winner_indices[i] = area.w + num_first_winners
				num_first_winners += 1
		area._new_winners = new_winner_indices
		area._new_w = area.w + num_first_winners

		# print name + " num_first_winners = " + str(num_first_winners)

		if verbose:
			print("new_winners: ")
			print(area._new_winners)

		# for i in num_first_winners
		# generate where input came from
			# 1) can sample input from array of size total_k, use ranges
			# 2) can use stars/stripes method: if m total inputs, sample (m-1) out of total_k
		first_winner_to_inputs = {}
		for i in range(num_first_winners):
			input_indices = random.sample(range(0, total_k), int(first_winner_inputs[i]))
			inputs = np.zeros(num_inputs)
			total_so_far = 0
			for j in range(num_inputs):
				inputs[j] = sum([((total_so_far + input_sizes[j]) > w >= total_so_far) for w in input_indices])
				total_so_far += input_sizes[j]
			first_winner_to_inputs[i] = inputs
			if verbose:
				print("for first_winner # " + str(i) + " with input " + str(first_winner_inputs[i]) + " split as so: ")
				print(inputs)

		m = 0
		# connectome for each stim->area
			# add num_first_winners cells, sampled input * (1+beta)
			# for i in repeat_winners, stimulus_inputs[i] *= (1+beta)
		for stim in from_stimuli:
			if num_first_winners > 0:
				self.stimuli_connectomes[stim][name] = np.resize(self.stimuli_connectomes[stim][name],
					area.w + num_first_winners)
			for i in range(num_first_winners):
				self.stimuli_connectomes[stim][name][area.w + i] = first_winner_to_inputs[i][m]
			stim_to_area_beta = area.stimulus_beta[stim]
			for i in area._new_winners:
				self.stimuli_connectomes[stim][name][i] *= (1+stim_to_area_beta)
			if verbose:
				print(stim + " now looks like: ") 
				print(self.stimuli_connectomes[stim][name])
			m += 1

		# connectome for each in_area->area
			# add num_first_winners columns
			# for each i in num_first_winners, fill in (1+beta) for chosen neurons
			# for each i in repeat_winners, for j in in_area.winners, connectome[j][i] *= (1+beta)
		for from_area in from_areas:
			from_area_w = self.areas[from_area].w
			from_area_winners = self.areas[from_area].winners
			self.connectomes[from_area][name] = np.pad(self.connectomes[from_area][name], 
				((0,0),(0,num_first_winners)), 'constant', constant_values=0)
			for i in range(num_first_winners):
				total_in = first_winner_to_inputs[i][m]
				sample_indices = random.sample(from_area_winners, int(total_in))
				for j in range(from_area_w):
					if j in sample_indices:
						self.connectomes[from_area][name][j][area.w+i] = 1
					if j not in from_area_winners:
						self.connectomes[from_area][name][j][area.w+i] = np.random.binomial(1,self.p)
			area_to_area_beta = area.area_beta[from_area]
			for i in area._new_winners:
				for j in from_area_winners:
					self.connectomes[from_area][name][j][i] *= (1.0 +area_to_area_beta)
			if verbose:
				print("Connectome of " + from_area + " to " + name + " is now:")
				print(self.connectomes[from_area][name])
			m += 1

		# expand connectomes from other areas that did not fire into area
		# also expand connectome for area->other_area
		for other_area in self.areas:
			if other_area not in from_areas:
				self.connectomes[other_area][name] = np.pad(self.connectomes[other_area][name], 
					((0,0),(0,num_first_winners)), 'constant', constant_values=0)
				for j in range(self.areas[other_area].w):
					for i in range(area.w, area._new_w):
						self.connectomes[other_area][name][j][i] = np.random.binomial(1, self.p)
			# add num_first_winners rows, all bernoulli with probability p
			self.connectomes[name][other_area] = np.pad(self.connectomes[name][other_area],
				((0, num_first_winners),(0, 0)), 'constant', constant_values=0)
			columns = len(self.connectomes[name][other_area][0])
			for i in range(area.w, area._new_w):
				for j in range(columns):
					self.connectomes[name][other_area][i][j] = np.random.binomial(1, self.p)
			if verbose:
				print("Connectome of " + name + " to " + other_area + " is now:")
				print(self.connectomes[name][other_area])

		return num_first_winners










	

