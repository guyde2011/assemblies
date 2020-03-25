""" Configurable brain assembly model for simulations and research.
Author: Daniel Mitropolsky, 2018

This module contains classes to represent different elements of a brain simulation:
    - Area - Represents an individual area of the brain, with the relevant parameters.
    - Connectomes are the connections between neurons. They have weights, which are initialized randomly but
        due to plasticity they can updated every time some neuron fires. These weights are represented by numpy arrays.
        The ones that are not random, because they were influenced by previous projections, are referred to as the 'support'.
    - Winners in a given 'round' are the specific neurons that fired in that round.
        In any specific area, these will be the 'k' neurons with the highest value flown into them.
        These are also the only neurons whose connectome weights get updated.
    - Stimulus - Represents a random stimulus that can be applied to any part of the brain.
        When a stimulus is created it is initialized randomly, but when applied multiple times this will change.
        This is equivalent to k neurons from an unknown part of the brain firing and their (initially, random)
        connectomes decide how this stimulus affects a given area of the brain.
    - Brain - A class representing a simulated brain, with it's different areas, stimulus, and all the connectome weights.
        A brain is initialized as a random graph, and it is maintained in a 'sparse' representation,
        meaning that all neurons that have their original, random connectome weights (0 or 1) are not saved explicitly.
    - Assembly - TODO define and express in code
"""
import logging
from typing import List, Mapping, Tuple, Dict, Any
import numpy as np
import heapq
from collections import defaultdict

from numpy.core._multiarray_umath import ndarray
from scipy.stats import binom
from scipy.stats import truncnorm
import math
import random


class Stimulus:
    """ Represents a random stimulus that can be applied to any part of the brain.
    That is, a specific set of k neurons that fire together that do not reside in
    any of the brain areas. These k neurons can be though of as representing a
    specific input stimulus.
    A stimulus can be connected to any areas and each pair of stimulus neuron and
    an area neuron initially have a synapse with probability p (Brain.p).

    The data for the synaptic weights is found in the downstream areas (the areas
    that this stimulus goes into).

    Attributes:
        k: number of neurons that fire
    """

    def __init__(self, k: int):
        self.k = k


class Area:
    """Represents an individual area of the brain.

    The list of neurons that are firing is given by 'winners'. It is updated through application of 'Brain.project',
    where the set of '_new_winners' is calculated and only updated once all brain areas settle on their new winners.
    The winners are the 'k' neurons with the highest value going in each round.

    Initially, most computation are represented implicitly. Winners are represented explicitly, and the changes in
    their incoming synapses are maintained in 'Brain.connectomes' and 'Brain.stimuli_connectomes'. The explicit neurons
    are represented by indices starting with 0 up to 'support_size'-1.

    Since it is initialized randomly, all the programmer needs to provide for initialization is the number 'n' of neurons,
    number 'k' of winners in any given round (meaning the k neurons with heights values will fire),
    and the parameter 'beta' of plasticity controlling connectome weight updates.

    TODO: remove '_new_winners'.
    TODO: remove 'name'. We prefer to use variable names to refer to areas.

    Attributes:
        n: number of neurons in this brain area
        k: number of winners in each round
        beta: plasticity parameter for self-connections
        stimulus_beta: plasticity parameters for connections from each incoming stimulus
        area_beta: plasticity parameters for connections from each incoming area
        support_size: The number of neurons that are represented explicitly (= total number of previous winners)
        winners: List of current winners. That is, 'k' top neurons from previous round.
        _new_support_size: the size of the support for the new update. Should be 'support_size' + 'num_first_winners'.
        _new_winners: During the projection process, a new set of winners is formed. The winners are only
            updated when the projection ends, so that the newly computed winners won't affect computation
        num_first_winners: should be equal to 'len(_new_winners)'
    """

    def __init__(self, name: str, n: int, k: int, beta: float = 0.05):
        self.name = name
        self.n = n
        self.k = k
        self.beta = beta
        self.stimulus_beta: Dict[str, float] = {}
        self.area_beta: Dict[str, float] = {}
        self.support_size: int = 0
        self.winners: List[int] = []
        self._new_support_size: int = 0
        self._new_winners: List[int] = []
        self.num_first_winners: int = -1

    def update_winners(self) -> None:
        """ This function updates the list of winners for this area after a projection step.

            TODO: redesign this so that the list of new winners is not saved in area.
        """
        self.winners = self._new_winners
        self.support_size = self._new_support_size


class Brain:
    """Represents a simulated brain, with it's different areas, stimuli, and all the synapse weights.

    The brain updates by selecting a subgraph of stimuli and areas, and activating only those connections.


    Attributes:
        areas: A mapping from area names to Area objects representing them.
        stimuli: A mapping from stimulus names to Stimulus objects representing them.
        stimuli_connectomes: Maps each pair of (stimulus,area) to the ndarray representing the synaptic weights among
            stimulus neurons and neurons in the support of area.
        connectomes: Maps each pair of areas to the ndarray representing the synaptic weights among neurons in
            the support.
        p: Probability of connectome (edge) existing between two neurons (vertices)
    """

    def __init__(self, p: float):
        self.areas: Dict[str, Area] = {}
        self.stimuli: Dict[str, Stimulus] = {}
        self.stimuli_connectomes: Dict[str, Dict[str, ndarray]] = {}
        self.connectomes: Dict[str, Dict[str, ndarray]] = {}
        self.p: float = p

    def add_stimulus(self, name: str, k: int) -> None:
        """ Initialize a random stimulus with 'k' neurons firing.
        This stimulus can later be applied to different areas of the brain,
        also updating its outgoing connectomes in the process.

        Connectomes to all areas is initialized as an empty numpy array.
        For every target area, which are all existing areas, set the plasticity coefficient, beta, to equal that area's beta.

        :param name: Name used to refer to stimulus
        :param k: Number of neurons in the stimulus
        """
        self.stimuli[name]: Stimulus = Stimulus(k)

        self.conectomes_init_stimulus(self.stimuli[name], name)

    def add_area(self, name: str, n: int, k: int, beta: float) -> None:
        """Add an area to this brain, randomly connected to all other areas and stimulus.

        Initialize each synapse weight to have a value of 0 or 1 with probability 'p'.
        Initialize incoming and outgoing connectomes as empty arrays.
        Initialize incoming betas as 'beta'.
        Initialize outgoing betas as the target area.beta

        :param name: Name of area
        :param n: Number of neurons in the new area
        :param k: Number of winners in the new area
        :param beta: plasticity parameter of connectomes coming INTO this area.
                The plastiity parameter of connectomes FROM this area INTO other areas are decided by
                the betas of those other areas.
        """
        self.areas[name] = Area(name, n, k, beta)

        # This should be replaced by conectomes_init_area(self, self.areas[name], beta).
        # (From here to the end of the function).
        self.conectomes_init_area(self.areas[name], beta)

    def project(self, stim_to_area: Mapping[str, List[str]],
                area_to_area: Mapping[str, List[str]]) -> None:
        """ Project is the basic operation where some stimuli and some areas are activated,
        with only specified connections between them active.

        :param stim_to_area: Dictionary that matches to each stimuli applied a list of areas to project into.
            Example: {"stim1":["A"], "stim2":["C","A"]}
        :param area_to_area: Dictionary that matches for each area a list of areas to project into.
            Note that an area can also be projected into itself.
            Example: {"A":["A","B"],"C":["C","A"]}
        """
        stim_in: defaultdict[str, List[str]] = defaultdict(lambda: [])
        area_in: defaultdict[str, List[str]] = defaultdict(lambda: [])

        # Validate stim_area, area_area well defined
        # Set stim_in to be the Dictionary that matches for every area the list of input stimuli.
        # Set areas_in to be the Dictionary that matches for every area the list of input areas.
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

        # to_update is the set of all areas that receive input
        to_update = set().union(list(stim_in.keys()), list(area_in.keys()))

        for area in to_update:
            num_first_winners = self.project_into(self.areas[area], stim_in[area], area_in[area])
            self.areas[area].num_first_winners = num_first_winners

        # once done everything, for each area in to_update: area.update_winners()
        for area in to_update:
            self.areas[area].update_winners()

    # Noam and Eden:
    def conectomes_init_area(self, area: Area, beta: float):
        # self.connectomes: Dict[str, Dict[str, ndarray]] = {}
        # self.connectomes[area.name][other_area] = neurons: ndarray (of size (area.n, other_area.n))
        # ndarray[i][j] = weight of connectome from neuron i (in area) to neuron j (in other area)
        name = area.name
        for stim_name, stim_connectomes in self.stimuli_connectomes.items():
            stimulus: Stimulus = self.stimuli[stim_name]
            stim_connectomes[name] = np.random.binomial(1, self.p, (stimulus.k, area.n)).astype(dtype='f')
            self.areas[name].stimulus_beta[stim_name] = beta

        new_connectomes: Dict[str, ndarray] = {}
        for other_area_name in self.areas:
            other_area: Area = self.areas[other_area_name]
            new_connectomes[other_area_name] = np.random.binomial(1, self.p, (area.n, other_area.n)).astype(dtype='f')
            if other_area_name != name:
                self.connectomes[other_area_name][name] = np.random.binomial(1, self.p, (other_area.n, area.n)).astype(dtype='f')
            self.areas[other_area_name].area_beta[name] = self.areas[other_area_name].beta
            self.areas[name].area_beta[other_area_name] = beta
        self.connectomes[name] = new_connectomes

    def conectomes_init_stimulus(self, stimulus: Stimulus, name: str):
        # self.stimuli_connectomes: Dict[str, Dict[str, ndarray]] = {}
        # self.connectomes[self.stimuli[name]][other_area] = neurons: ndarray (of size (stimuli.k, other_area.n))
        # ndarray[i][j] = weight of connectome from neuron i (in stimulus) to neuron j (in other area)

        new_connectomes: Dict[str, ndarray] = {}
        for key in self.areas:
            other_area: Area = self.areas[key]
            new_connectomes[key] = np.random.binomial(1, self.p, (stimulus.k, other_area.n)).astype(dtype='f')
            self.areas[key].stimulus_beta[name] = self.areas[key].beta
        self.stimuli_connectomes[name] = new_connectomes

    # All of the following will be inside project_into
    # Guy
    def project_into_calculate_inputs(self, area: Area, from_stimuli: List[str], from_areas: List[str]):
        prev_winner_inputs = np.zeros(area.n)

        if from_areas:
            prev_winner_inputs += sum([np.dot(np.ones(self.areas[other_area].n), self.connectomes[other_area][area.name]) for other_area in from_areas])

        if from_stimuli:
             prev_winner_inputs += sum([np.dot(np.ones(self.stimuli[stim].k), self.stimuli_connectomes[stim][area.name]) for stim in from_stimuli])
        logging.debug(f'prev_winner_inputs: {prev_winner_inputs}')
        return prev_winner_inputs

    # Shai:
    def project_into_calculate_winners(self, area: Area, inputs):
        area._new_winners = heapq.nlargest(area.k, list(range(len(inputs))), inputs.__getitem__)
        logging.debug(f'new_winners: {area._new_winners}')

    # Adi
    def project_into_update_conectomes(self, area: Area, from_stimuli: List[str], from_areas: List[str]):
        # connectome for each stim->area
        # for i in new_winners, stimulus_inputs[i] *= (1+beta)
        for stim in from_stimuli:
            beta = area.stimulus_beta[stim]
            for i in area._new_winners:
                for j in range(self.stimuli[stim].k):
                    self.stimuli_connectomes[stim][area.name][j][i] *= (1 + beta)
            logging.debug(f'stimulus {stim} now looks like: {self.stimuli_connectomes[stim][area.name]}')

        # connectome for each in_area->area
        # for each i in _new_winners, for j in in_area.winners, connectome[j][i] *= (1+beta)
        for from_area in from_areas:
            from_area_winners = self.areas[from_area].winners
            beta = area.area_beta[from_area]
            # connectomes of winners are now stronger
            for i in area._new_winners:
                for j in from_area_winners:
                    self.connectomes[from_area][area.name][j][i] *= (1 + beta)
            logging.debug(f'Connectome of {from_area} to {area.name} is now {self.connectomes[from_area][area.name]}')
        return 0

    def project_into_non_lazy(self, area: Area, from_stimuli: List[str], from_areas: List[str]):
        inputs = self.project_into_calculate_inputs(area, from_stimuli, from_areas)
        self.project_into_calculate_winners(area, inputs)
        return self.project_into_update_conectomes(area, from_stimuli, from_areas)

    def project_into(self, area: Area, from_stimuli: List[str], from_areas: List[str]) -> int:
        """Project multiple stimuli and area assemblies into area 'area' at the same time.

        :param area: The area projected into
        :param from_stimuli: The stimuli that we will be applying
        :param from_areas: List of separate areas whose assemblies we will projected into this area
        :return: Returns the number of area neurons that were winners for the first time during this projection
        """
        # projecting everything in from stim_in[area] and area_in[area]
        # calculate: inputs to self.connectomes[area] (previous winners)
        # calculate: potential new winners, Binomial(sum of in sizes, k-top)
        # k top of previous winners and potential new winners
        # if new winners > 0, redo connectome and intra_connectomes
        # have to wait to replace new_winners
        # TODO Add more documentation to this function which does most of the work
        # TODO Handle case of projecting from an area without previous winners.
        # TODO: there is a bug when adding a new stimulus later on.
        # TODO: Stimulus is updating to somehow represent >100 neurons.
        return self.project_into_non_lazy(area, from_stimuli, from_areas)
