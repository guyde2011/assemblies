from itertools import chain
from numpy.core import ndarray
from typing import Dict, List, Iterable
import numpy as np
from collections import defaultdict

# TODO: please use absolute and not relative imports. they will be clearer and easier to maintain
from ..performance import RandomMatrix

from ..components import Area, BrainPart, Stimulus, Connection
from .abstract_connectome import AbstractConnectome


class Connectome(AbstractConnectome):
    # TODO 2: fix documentation
    """
    Implementation of a random based connectome, based on the abstract connectome.
    The object representing the connection in here is ndarray from numpy
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

        self.rng = RandomMatrix()
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
        synapses = self.rng.multi_generate(area.n, part.n, self.p).reshape((part.n, area.n), order='F')

        self.connections[part, area] = Connection(part, area, synapses)

    # @TuringMachine: Look into this
    # TODO: `Dict[BrainPart, List[Area]]` is a complex type that should be defined by name
    # TODO 2: document
    def subconnectome(self, connections: Dict[BrainPart, List[Area]]) -> AbstractConnectome:
        # TODO: split the following line to two small methods with indicative names
        # TODO 2: can `areas` and `stimuli` be treated together?
        areas = set([part for part in connections if isinstance(part, Area)] + list(chain(*connections.values())))
        stimuli = [part for part in connections if isinstance(part, Stimulus)]
        edges = [(part, area) for part in connections for area in connections[part]]
        neural_subnet = [(edge, self.connections[edge]) for edge in edges]
        nlc = Connectome(self.p, areas=list(areas), stimuli=stimuli, connections=neural_subnet,
                         initialize=False)
        return nlc
        # TODO fix this, this part doesn't work with the new connections implemnentation!

    def get_connected_parts(self, area: Area) -> List[BrainPart]:
        return [source for source, dest in self.connections if dest == area]

    def update_connectomes(self, new_winners: Dict[Area, List[int]], sources: Dict[Area, List[BrainPart]]) -> None:
        """
        Update the connectomes of the areas with new winners, based on the plasticity.
        :param new_winners: the new winners per area
        :param sources: the sources of each area
        """
        if self._plasticity_disabled:
            return
        for area in new_winners:
            for source in sources[area]:
                # TODO: it seems that `beta` should be a property of the relation, rather than dynamically computed here
                # TODONT: beta isn't really a property of any single brain part, and the parts of a brain are dynamic,
                # so in reality there isn't really any place to keep this property
                # TODO: a concept of "relation" should be defined, and be responsible for this computation.
                # TODO: the current approach is problematic - you have to remember to check the type of the parameter
                # TODO: everywhere. it is prone to bugs and wrong assumptions. the code using `source` should not be
                # TODO: checking its type everywhere
                beta = source.beta if isinstance(source, Area) else area.beta
                source_neurons: Iterable[int] = \
                    range(source.n) if isinstance(source, Stimulus) else self.winners[source]
                # TODO: extract to small function, document the use of numpy vectorization to avoid misuse in the future
                # TODONT: Extracting this piece of code to a different function would require:
                # A) A lot of parameters (source, area, source_neurons, new_winners, beta) for a single line function
                # B) Will most likely cause more misuse, as giving this line a whole function means it can be called
                # Without invoking the whole logic
                # TODONT reply: you don't have to pass every single variable as a parameter. the function can accept a compound object
                # TODONT reply: (even the whole left hand side of the line)
                # TODONT reply: the purpose is to give the operation a name. it will help avoid misuse
                # TODO 2: is it possible to improve performance here? (is the iterable utilized correctly or can be changed?)
                # TODONT: This is one of the lines we've worked on the most to try different variations, no it isn't
                # possible but I DARE you to try
                self.connections[source, area].synapses[source_neurons, new_winners[area][:, None]] *= (1 + beta)

    def update_winners(self, new_winners: Dict[Area, List[int]], sources: Dict[Area, List[BrainPart]]) -> None:
        """
        Update the winners of areas with new winners.
        :param new_winners: the new winners per area
        :param sources: the sources of each area
        """
        to_update = sources.keys()
        for area in to_update:
            self.winners[area] = new_winners[area]

    def _project_into(self, area: Area, sources: List[BrainPart]) -> List[int]:
        """Project multiple stimuli and area assemblies into area 'area' at the same time.
        :param area: The area projected into
        :param sources: List of separate brain parts whose assemblies we will projected into this area
        :return: Returns new winners of the area
        """
        # Calculate the total input for each neuron from other given areas' winners and given stimuli.
        # Said total inputs list is saved in prev_winner_inputs
        # TODO: can you combine stimuli and areas to avoid writing logic twice?
        # TODONT: We used to do that, but for optimization we need to the logic a bit differently twice
        src_areas = [src for src in sources if isinstance(src, Area)]
        src_stimuli = [src for src in sources if isinstance(src, Stimulus)]
        for part in sources:
            if (part, area) not in self.connections:
                self._initialize_connection(part, area)

        prev_winner_inputs: ndarray = np.zeros(area.n)
        for source in src_areas:
            area_connectome = self.connections[source, area]
            # TODO: can performance be improved using slicing?
            # TODONT: We've tried that, slicing here takes a lot more time
            prev_winner_inputs += np.sum((area_connectome.synapses[winner, :] for winner in self.winners[source]), axis=0)
        if src_stimuli:
            prev_winner_inputs += np.sum(self.connections[stim, area].synapses.sum(axis=0) for stim in src_stimuli)
        return np.argpartition(prev_winner_inputs, area.n - area.k)[-area.k:]

    # TODO: change name
    # TODONT: TO WHAT???!!!!!
    # TODO: something like `step` or `fire` that does not conflict with other names
    def project(self, connections: Dict[BrainPart, List[Area]]):
        """ Project is the basic operation where some stimuli and some areas are activated,
        with only specified connections between them active.
        :param connections A dictionary of connections to use in the projection, for example {area1
        """
        # TODO: is the defaultdict needed? (it seems `sources_mapping` values are initialized anyways)
        # TODONT: assigning a base value of empty list to each entry takes more lines
        # TODO: but this is what already happens in the code
        sources_mapping: defaultdict[Area, List[BrainPart]] = defaultdict(lambda: [])

        for part, areas in connections.items():
            for area in areas:
                if area not in sources_mapping:
                    sources_mapping[area] = []
                sources_mapping[area] = sources_mapping[area] or []
                sources_mapping[area].append(part)
        # to_update is the set of all areas that receive input
        to_update = sources_mapping.keys()

        new_winners: Dict[Area, List[int]] = dict()
        for area in to_update:
            new_winners[area] = self._project_into(area, sources_mapping[area])

        self.update_connectomes(new_winners, sources_mapping)
        self.update_winners(new_winners, sources_mapping)
