from itertools import chain
from numpy.core import ndarray
from typing import Dict, List, Iterable, NamedTuple, cast
import numpy as np
from collections import defaultdict

from ..Performance import MultithreadedRNG
from ..Performance.multithreaded.multi_sum import multi_sum

from ..components import Area, BrainPart, Stimulus, Connection
from .abc_connectome import ABCConnectome



class Connectome(ABCConnectome):
    """
    Implementation of Non lazy random based connectome, based on the generic connectome.
    The object representing the connection in here is ndarray from numpy

    Attributes:
        (All the attributes of Connectome
        p: The probability for each edge of the connectome to exist
        initialize: Whether or not to fill the connectome of the brain in each place the connections are missing. If
        this is a subconnectome the initialize flag should be False
    """
    def __init__(self, p: float, areas=None, stimuli=None, connections=None, initialize=False):
        """
        :param p: The attribute p for the probability of an edge to exits
        :param areas: list of areas
        :param stimuli: list of stimuli
        :param connections: Optional argument which gives active connections to the connectome
        :param initialize: Whether or not to initialize the connectome of the brain.
        """
        super(Connectome, self).__init__(p, areas, stimuli)

        self.rng = MultithreadedRNG()
        self._winners: Dict[Area, List[int]] = defaultdict(lambda: [])
        if initialize:
            self._initialize_parts((areas or []) + (stimuli or []))

    def add_area(self, area: Area):
        super().add_area(area)
        self._initialize_parts([area])

    def add_stimulus(self, stimulus: Stimulus):
        super().add_stimulus(stimulus)
        self._initialize_parts([stimulus])

    def _set_winners(self, area: Area, winners: List[int]):
        self._winners[area] = winners

    def _get_winners(self, area: Area) -> List[int]:
        return self._winners[area]

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

        synapses = self.rng.multi_generate(part.n, area.n, self.p)
        # synapses = np.random.binomial(1, self.p, (part.n, area.n)).astype(dtype='f')
        # synapses = sp.random(part.n, area.n, density=self.p, data_rvs=np.ones, format='lil', dtype='f')
        self.connections[part, area] = Connection(part, area, synapses)

    def subconnectome(self, connections: Dict[BrainPart, List[Area]]) -> ABCConnectome:
        areas = set([part for part in connections if isinstance(part, Area)] + list(chain(*connections.values())))
        stimuli = [part for part in connections if isinstance(part, Stimulus)]
        edges = [(part, area) for part in connections for area in connections[part]]
        neural_subnet = [(edge, self.connections[edge]) for edge in edges]
        nlc = Connectome(self.p, areas=list(areas), stimuli=stimuli, connections=neural_subnet,
                         initialize=False)
        return nlc
        # TODO fix this, this part doesn't work with the new connections implemnentation!

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
                source_neurons: Iterable[int] = \
                    range(source.n) if isinstance(source, Stimulus) else list(self.winners[source])

                self.connections[source, area].synapses[source_neurons, np.asarray(new_winners[area])[:, None]] *= (1 + beta)

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
        for part in sources:
            if (part, area) not in self.connections:
                self._initialize_connection(part, area)

        prev_winner_inputs: ndarray = np.zeros(area.n)
        for source in src_areas:
            area_connectomes = self.connections[source, area]
            prev_winner_inputs += \
                multi_sum((area_connectomes.synapses), self.winners[source])
        if src_stimuli:
            prev_winner_inputs += sum(self.connections[stim, area].synapses.sum(axis=0) for stim in src_stimuli)
        return np.argpartition(prev_winner_inputs, area.k-1)[-area.k:]

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

        self.update_connectomes(new_winners, sources_mapping)

        # once done everything, update areas winners
        for area in to_update:
            self.winners[area] = new_winners[area]
