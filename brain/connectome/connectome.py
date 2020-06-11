from __future__ import annotations  # import annotations from later version of python.
# We need it here to annadiane that connectome has a method which returns itself

from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple, Set, TypeVar, Mapping, Optional, Iterator, Iterable

from wrapt import ObjectProxy  # Needed to pip install
from functools import cached_property
from collections import defaultdict

from ..components import BrainPart, Area, Stimulus, Connection



# The wrapt library implements easy to use wrapper objects, which delegates everything to the object you are
# using. It's very convenient to use (it can be used exactly in the same way).
# More info and examples:
# https://wrapt.readthedocs.io/en/latest/wrappers.html


K_co = TypeVar('K_co', covariant=True)
V_contra = TypeVar('V_contra', contravariant=True)


class MappingProxy(Mapping[K_co, V_contra]):
    def __init__(self, mapping: Mapping[K_co, V_contra]):
        self._mapping: Mapping[K_co, V_contra] = mapping

    def __getitem__(self, key: K_co) -> Optional[V_contra]:
        return self._mapping[key]

    def get(self, key: K_co, *, default: Optional[V_contra]=None):
        return self[key] if key in self._mapping else default

    def __len__(self) -> int:
        return len(self._mapping)

    def __iter__(self) -> Iterator[K_co]:
        return iter(self._mapping)


class Connectome(metaclass=ABCMeta):
    """
    Represent the graph of connections between areas and stimuli of the brain.
    This is a generic abstract class which offer a good infrastructure for building new models of connectome.
    You should implement some of the parts which are left for private case. For example when and how the connectome
    should be initialized, how the connections are represented.
    Attributes:
        areas: List of area objects in the connectome
        stimuli: List of stimulus objects in the connectome
        connections: Dictionary from tuples of BrainPart(Stimulus/Area) and Area to some object which
        represent the connection (e.g. numpy matrix). Each connection is held ObjectProxy which will
        make the connection.
        to be saved by reference. (This makes the get_subconnectome routine much easier to implement)
    """

    def __init__(self, p, areas=None, stimuli=None):
        self.areas: List[Area] = []
        self.stimuli: List[Stimulus] = []
        self.connections: Dict[Tuple[BrainPart, Area], Connection] = {}
        self.p = p

        self._winners: Dict[Area, Set[int]] = defaultdict(lambda: set())
        # TODO add support
        if areas:
            self.areas = areas
        if stimuli:
            self.stimuli = stimuli

    def free_memory(self):
        pass

    def add_area(self, area: Area):
        self.areas.append(area)

    def add_stimulus(self, stimulus: Stimulus):
        self.stimuli.append(stimulus)

    @cached_property
    def winners(self) -> Mapping[Area, Set[int]]:
        return MappingProxy(self._winners)

    @abstractmethod
    def subconnectome(self, connections: Dict[BrainPart, Area]) -> Connectome:
        """
        Retrieve restriction of the connectome to specific subconnectome.
        Note that changes to the returned subconnectome should reflect in the original one. (By reference)
        :param connections: directed connections needed in the subconnectome
        :return: A connectome which is a subgraph of the self connectome, according to the mapping in connections
        """
        pass

    @abstractmethod
    def area_connections(self, area: Area) -> List[BrainPart]:
        """
        Retrieve all parts with connection to specific areas, according to the current connectome
        :param area: area which we need the connections to
        :return: List of all connections to the area
        """
        pass

    @abstractmethod
    def project(self, connections: Dict[BrainPart, Iterable[Area]]):
        """
        Project is the basic operation where some stimuli and some areas are activated,
        with only specified connections between them active.
        :param connections: A dictionary of connections to use in the projection, for example {area1
        """
        pass

    def __repr__(self):
        return f'{self.__class__.__name__} with {len(self.areas)} areas, and {len(self.stimuli)} stimuli'