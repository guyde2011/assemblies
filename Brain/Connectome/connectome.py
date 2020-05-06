from __future__ import annotations  # import annotations from later version of python.
# We need it here to annadiane that connectome has a method which returns itself

from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple

from wrapt import ObjectProxy  # Needed to pip install

from ..components import BrainPart, Area, Stimulus, Connection


# The wrapt library implements easy to use wrapper objects, which delegates everything to the object you are
# using. It's very convenient to use (it can be used exactly in the same way).
# More info and examples:
# https://wrapt.readthedocs.io/en/latest/wrappers.html


class Connectome:
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
    __metaclass__ = ABCMeta  # This is some magic code which makes the connectome become an abstract class

    def __init__(self, p, areas=None, stimuli=None):
        self.areas: List[Area] = []
        self.stimuli: List[Stimulus] = []
        self.connections: Dict[Tuple[BrainPart, Area], Connection] = {}
        self.p = p

        if areas:
            self.areas = areas
        if stimuli:
            self.stimuli = stimuli

    def add_area(self, area: Area):
        self.areas.append(area)

    def add_stimulus(self, stimulus: Stimulus):
        self.stimuli.appennd(stimulus)

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

    def __repr__(self):
        return f'{self.__class__.__name__} with {len(self.areas)} areas, and {len(self.stimuli)} stimuli'
