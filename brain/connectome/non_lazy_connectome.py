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

    def __init__(self, p: float, areas=None, stimuli=None, connections=None, initialize=True):
        """
        :param p: The attribute p for the probability of an edge to exits
        :param areas: list of areas
        :param stimuli: list of stimuli
        :param connections: Optional argument which gives active connections to the connectome
        :param initialize: Whether or not to initialize the connectome of the brain.
        """
        super(NonLazyConnectome, self).__init__(p, areas, stimuli)

        if initialize:
            self._initialize_parts((areas or []) + (stimuli or []))

    def add_area(self, area: Area):
        super().add_area(area)
        self._initialize_parts([area])

    def add_stimulus(self, stimulus: Stimulus):
        super().add_stimulus(stimulus)
        self._initialize_parts([stimulus])

    def _initialize_parts(self, parts: List[BrainPart]):
        """
        Initialize all the connections to and from the given brain parts.
        :param parts: List of stimuli and areas to initialize
        :return:
        """
        for part in parts:
            for other in self.areas + self.stimuli:
                self._initialize_connection(part, other)
                if isinstance(part, Area) and part != other:
                    self._initialize_connection(other, part)

    def _initialize_connection(self, part: BrainPart, area: Area):
        """
        Initalize the connection from brain part to an area
        :param part: Stimulus or Area which the connection should come from
        :param area: Area which the connection go to
        :return:
        """
        synapses = np.random.binomial(1, self.p, (part.n, area.n)).astype(dtype='f')
        self.connections[part, area] = Connection(part, area, synapses)

    def subconnectome(self, connections: Dict[BrainPart, List[Area]]) -> Connectome:
        areas = set([part for part in connections if isinstance(part, Area)] + list(chain(*connections.values())))
        stimuli = [part for part in connections if isinstance(part, Stimulus)]
        edges = [(part, area) for part in connections for area in connections[part]]
        neural_subnet = [(edge, self.connections[edge]) for edge in edges]
        nlc = NonLazyConnectome(self.p, areas=list(areas), stimuli=stimuli, connections=neural_subnet,
                                initialize=False)
        return nlc
        # TODO fix this, this part doesn't work with the new connections implemnentation!
        # nlc =

    def area_connections(self, area: Area) -> List[BrainPart]:
        return [source for source, dest in self.connections if dest == area]

    def update_connectomes(self, new_winners: Dict[Area, List[int]], sources: Dict[Area, List[BrainPart]]) -> None:
        """
        Update the connectomes of the areas with new winners, based on the plasticity.
        :param new_winners: the new winners per area
        :param sources: the sources of each area
        """
        for area in new_winners:
            for source in sources[area]:
                beta = source.beta if isinstance(source, Area) else area.beta
                for i in new_winners[area]:
                    # update weight (*(1+beta)) for all neurons in stimulus / the winners in area
                    source_neurons: List[int] = range(source.n) if isinstance(source, Stimulus) else source.winners
                    for j in source_neurons:
                        self.connections[source, area][j][i] *= (1 + beta)
                print(f'connection {source}-{area} now looks like: {self.connections[source, area]}')

    def project_into(self, area: Area, sources: List[BrainPart]) -> List[int]:
        """Project multiple stimuli and area assemblies into area 'area' at the same time.
        :param area: The area projected into
        :param sources: List of separate brain parts whose assemblies we will projected into this area
        :return: Returns new winners of the area
        """
        # Calculate the total input for each neuron from other given areas' winners and given stimuli.
        # Said total inputs list is saved in prev_winner_inputs
        src_areas = [src for src in sources if isinstance(src, Area)]
        src_stimuli = [src for src in sources if isinstance(src, Stimulus)]

        prev_winner_inputs: List[float] = np.zeros(area.n)
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
                    self.connections[stim, area].synapses
                )
                for stim in src_stimuli
            ])

        print(f'prev_winner_inputs: {prev_winner_inputs}')
        return heapq.nlargest(area.k, list(range(len(prev_winner_inputs))), prev_winner_inputs.__getitem__)

    def project(self, connections: Dict[BrainPart, List[Area]]):
        """ Project is the basic operation where some stimuli and some areas are activated,
        with only specified connections between them active.
        :param connections A dictionary of connections to use in the projection, for example {area1
        """
        sources_mapping: defaultdict[Area, List[BrainPart]] = defaultdict(lambda: [])

        for part, areas in connections.items():
            for area in areas:
                sources_mapping[area] = sources_mapping[area] or []
                sources_mapping[area].append(part)
        # to_update is the set of all areas that receive input
        to_update = sources_mapping.keys()

        new_winners: Dict[Area, List[int]] = dict()
        for area in to_update:
            new_winners[area] = self.project_into(area, sources_mapping[area])
            print(f'new winners of {area}: {new_winners[area]}')

        self.update_connectomes(new_winners, sources_mapping)

        # once done everything, update areas winners
        for area in to_update:
            area.winners = new_winners[area]
