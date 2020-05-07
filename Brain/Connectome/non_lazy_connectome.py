from itertools import product, chain
from numpy.core._multiarray_umath import ndarray
from typing import Dict, List, Tuple
import logging
import heapq
import numpy as np
from collections import defaultdict
from wrapt import ObjectProxy

from ..components import Area, BrainPart, Stimulus, Connection
from .connectome import Connectome


class NonLazyConnectome(Connectome):
    """
    Implementation of Non lazy random based connectome, based on the generic connectome.
    The object representing the connection in here is ndarray from numpy

    Attributes:
        (All the attributes of Connectome
        p: The probability for each edge of the connectome to exist
        initialize: Whether or not to fill the connectome of the brain in each place the connections are missing. If
        this is a subconnectome the initialize flag should be False
    """

    def __init__(self, p: float, brainparts: List[BrainPart]=None, connections: Dict[(BrainPart, Brain), Connection]=None, initialize=True):
        """
        :param p: The attribute p for the probability of an edge to exits
        :param areas: list of areas
        :param stimuli: list of stimuli
        :param connections: Optional argument which gives active connections to the connectome
        :param initialize: Whether or not to initialize the connectome of the brain.
        """
        super(NonLazyConnectome, self).__init__(brainparts, connections)
        self.p = p
        self.active = connections
        if initialize:
            self._initialize_parts(brainparts)

    def inhibit(self, parts: List[(BrainPart, BrainPart)]):
        for couple in parts:
            self.active[couple] = self.connections[couple]

    def disinhibit(self, parts: List[(BrainPart, BrainPart)]):
        for couple in parts:
            try:
                del self.active[couple]
            except KeyError:
                continue

            
    def add_brain_part(self, brainpart: BrainPart):
        self.brain_parts.append(brainpart)
        self._initialize_parts([brainpart], self.active)

    def _initialize_parts(self, parts: List[BrainPart], new_connections: Dict[BrainPart, List[BrainPart]):
        """
        Initialize all the connections to and from the given brain parts.
        :param parts: List of stimuli and areas to initialize
        :return:
        """
        for part in parts:
            others = new_connections[part]

            if part.part_type == 'Area':
                for other in filer(lambda p: p.part_type in ['Area', 'OutputArea'], others):
                    self._initialize_connection(part, other)
                for other in filer(lambda p: p.part_type in ['Area', 'Stimulus'] and p != part, others):
                    self._initialize_connection(other, part)

            if part.part_type == 'stimulus':
                for other in filer(lambda p: p.part_type in ['Area'], others):
                    self._initialize_connection(part, other)

            if part.part_type == 'OutputArea':
                for other in filer(lambda p: p.part_type in ['Area'], others):
                    self._initialize_connection(other, part)


    def _initialize_connection(self, part: BrainPart, other: BrainPart):
        """
        Initalize the connection from brain part to an area
        :param part: Stimulus or Area which the connection should come from
        :param area: Area which the connection go to
        :return:
        """
        synapses = np.random.binomial(1, self.p, (part.n, area.n)).astype(dtype='f')
        self.connections[part, other] = Connection(part, other, synapses)


    def _update_connectomes(self, new_winners: Dict[BrainPart, List[int]], sources: Dict[Area, List[BrainPart]]) -> None:
        """
        Update the connectomes of the areas with new winners, based on the plasticity.
        :param new_winners: the new winners per area
        :param sources: the sources of each area
        """
        for part in new_winners:
            for source in sources[part]:
                beta = source.beta if source.part_type='Area' else part.beta
                for i in new_winners[part]:
                    # update weight (*(1+beta)) for all neurons in stimulus / the winners in area
                    source_neurons: List[int] = range(source.n) if source.part_type='Stimulus' else source.winners
                    for j in source_neurons:
                        self.connections[source, part][j][i] *= (1 + beta)
                print(f'connection {source}-{area} now looks like: {self.connections[source, area]}')

    def project_into(self, part: BrainPart, sources: List[BrainPart]) -> List[int]:
        """Project multiple stimuli and area assemblies into area 'area' at the same time.
        :param area: The area projected into
        :param sources: List of separate brain parts whose assemblies we will projected into this area
        :return: Returns new winners of the area
        """
        # Calculate the total input for each neuron from other given areas' winners and given stimuli.
        # Said total inputs list is saved in prev_winner_inputs
        src_areas = [src for src in sources if src.part_type='Area')]
        src_stimuli = [src for src in sources if src='Stimulus')]

        prev_winner_inputs: List[float] = np.zeros(part.n)
        for source in src_areas:
            area_connectomes = self.connections[source, area]
            for winner in source.winners:
                prev_winner_inputs += area_connectomes[winner]

        if src_stimuli:
            prev_winner_inputs += sum([
                np.dot(
                    np.ones(
                        stim.n
                    ),
                    self.connections[stim, part].synapses
                )
                for stim in src_stimuli
            ])

        print(f'prev_winner_inputs: {prev_winner_inputs}')
        return heapq.nlargest(part.k, list(range(len(prev_winner_inputs))), prev_winner_inputs.__getitem__)

    def project(self, connections: Dict[BrainPart, List[BrainPart]]):
        """ Project is the basic operation where some stimuli and some areas are activated,
        with only specified connections between them active.
        :param connections A dictionary of connections to use in the projection, for example {area1
        """
        sources_mapping: defaultdict[BrainPart, List[BrainPart]] = defaultdict(lambda: [])

        for suorce, destinations in connections.items():
            for dest in destinations:
                sources_mapping[dest] = sources_mapping[dest] or []
                sources_mapping[dest].append(source)
        # to_update is the set of all areas that receive input
        to_update = sources_mapping.keys()

        new_winners: Dict[BrainPart, List[int]] = dict()
        for part in to_update:
            new_winners[part] = self.project_into(part, sources_mapping[part])
            print(f'new winners of {part}: {new_winners[part]}')

        self.update_connectomes(new_winners, sources_mapping)

        # once done everything, update areas winners and support
        for part in to_update:
            part.winners = new_winners[part]
            for winner in new_winners[part]:
                part.support[winner] = 1
