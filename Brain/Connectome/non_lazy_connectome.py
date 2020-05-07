from collections import defaultdict
import heapq
from typing import Dict, List

import numpy as np

from .components import BrainPart, Connection
from .connectome import Connectome


class NonLazyConnectome(Connectome):
    """
    Implementation of Non lazy random based connectome, based on the generic connectome.
    The object representing the connection in here is ndarray from numpy

    Attributes:
        (All the attributes of Connectome)
        p: The probability for each edge of the connectome to exist
    """

    def __init__(self, p: float, brain_parts: List[BrainPart] = None,
                 connections: Dict[(BrainPart, BrainPart), Connection] = None, initialize=True):
        """
        :param p: The attribute p for the probability of an edge to exits
		:param brain_parts: Optional - Initialize list of brain parts which constructs the connectome.
        :param connections: Optional argument which gives active connections to the connectome
        :param initialize: Whether or not to initialize the connectome of the brain.
        """
        super(NonLazyConnectome, self).__init__(brainparts, connections)
        self.p = p
        if initialize:
            self._initialize_parts(brainparts)

    def add_brain_part(self, brainpart: BrainPart):
        self.brain_parts.append(brainpart)
        self._initialize_parts([brainpart])

    def _initialize_parts(self, brain_parts: List[BrainPart]):
        """
        Initialize all the connections to and from the given brain parts.
        :param brain_parts: List of stimuli and areas to initialize.
        """

        for part_type in ['Area', 'Stimulus', 'OutputArea']:
            parts = [part for part in filter(lambda p: p.part_type == part_type, self.brain_parts)]
            if part_type == 'Area':
                # connections from Areas
                others = [other for other in filter(lambda p: p.part_type in ['Area', 'OutputArea'], self.brain_parts)]
                self._initialize_connections({part: others for part in parts})
                # connections to Area
                others = [other for other in filter(lambda p: p.part_type in ['Area', 'Stimulus'], self.brain_parts)]
                self._initialize_connections({other: parts for other in others})

            if part_type == 'stimulus':
                # connections from stimuli
                others = [other for other in filter(lambda p: p.part_type in ['Area'], self.brain_parts)]
                self._initialize_connections({part: others for part in parts})

            if part_type == 'OutputArea':
                # connections to output areas
                others = [other for other in filter(lambda p: p.part_type in ['Area'], self.brain_parts)]
                self._initialize_connection({other: parts for other in others})

    def _initialize_connections(self, new_connections: Dict[BrainPart, List[BrainPart]]):
        """
        Initialize several connections from specific parts to others.
        :param new_connections: Dict of source brain parts to a destination brain part, which should be initialized.
        """
        for source in new_connections:
            for dest in new_connections[source]:
                self._initialize_connection(source, dest)

    def _initialize_connection(self, part: BrainPart, other: BrainPart):
        """
        Initialize the connection from brain part to an area
        :param part: Stimulus or Area which the connection should come from
        :param area: Area which the connection go to
        :return:
        """
        if (part, other) in self.connections.keys():
            return
        synapses = np.random.binomial(1, self.p, (part.n, part.n)).astype(dtype='f')
        self.connections[part, other] = Connection(part, other, synapses)

    def _update_connectomes(self, new_winners: Dict[BrainPart, List[int]],
                            sources: Dict[BrainPart, List[BrainPart]]) -> None:
        """
        Update the connectomes of the areas with new winners, based on the plasticity.
        :param new_winners: the new winners per area
        :param sources: the sources of each area
        """
        for part in new_winners:
            for source in sources[part]:
                beta = source.beta if source.part_type == 'Area' else part.beta
                for i in new_winners[part]:
                    # update weight (*(1+beta)) for all neurons in stimulus / the winners in area
                    source_neurons: List[int] = range(source.n) if source.part_type == 'Stimulus' else source.winners
                    for j in source_neurons:
                        self.connections[source, part][j][i] *= (1 + beta)
                print(f'connection {source}-{part} now looks like: {self.connections[source, part]}')

    def project_into(self, part: BrainPart, sources: List[BrainPart]) -> List[int]:
        """
		Project multiple stimuli and area assemblies into area 'area' at the same time.
        :param area: The area projected into
        :param sources: List of separate brain parts whose assemblies we will projected into this area
        :return: Returns new winners of the area
        """
        # Calculate the total input for each neuron from other given areas' winners and given stimuli.
        # Said total inputs list is saved in prev_winner_inputs
        src_areas = [src for src in sources if src.part_type == 'Area']
        src_stimuli = [src for src in sources if src.part_type == 'Stimulus']

        prev_winner_inputs: List[float] = np.zeros(part.n)
        for source in src_areas:
            area_connectomes = self.connections[source, part]
        for winner in source.winners:
            prev_winner_inputs += area_connectomes[winner]

        if src_stimuli:
            prev_winner_inputs += sum(
                [np.dot(p.ones(stim.n), self.connections[stim, part].synapses) for stim in src_stimuli])

        print(f'prev_winner_inputs: {prev_winner_inputs}')

        return heapq.nlargest(part.k, list(range(len(prev_winner_inputs))), prev_winner_inputs.__getitem__)

    def next_round(self, connections: Dict[BrainPart, List[BrainPart]]):
        """ 
		Project is the basic operation where some stimuli and some areas are activated,
        with only specified connections between them active.
        :param connections: A dictionary of connections to use in the projection, for example {area1
        """
        sources_mapping: defaultdict[BrainPart, List[BrainPart]] = defaultdict(lambda: [])

        for source, destinations in connections.items():
            for dest in destinations:
                sources_mapping[dest] = sources_mapping[dest] or []
                sources_mapping[dest].append(source)
        # to_update is the set of all areas that receive input
        to_update = sources_mapping.keys()

        new_winners: Dict[BrainPart, List[int]] = dict()
        for part in to_update:
            new_winners[part] = self.project_into(part, sources_mapping[part])
            print(f'new winners of {part}: {new_winners[part]}')

        self._update_connectomes(new_winners, sources_mapping)

        # once done everything, update areas winners and support
        for part in to_update:
            part.winners = new_winners[part]
            for winner in new_winners[part]:
                part.support[winner] = 1
