from functools import wraps
from inspect import Parameter

from assemblies.argument_manipulation import argument_restrict, argument_extend
from ..brain.brain import *
from typing import Iterable, Union, Optional
from copy import deepcopy

Projectable = Union['Assembly', 'NamedStimulus']


class NamedStimulus(object):  # hi
    """ 
    acts as a buffer between our implementation and brain.py, as the relevant
    functions there use naming to differentiate between areas
    """

    def __init__(self, name, stimulus):
        self.name = name
        self.stimulus = stimulus

    def __repr__(self) -> str:
        return f"Stimulus({self.name})"


def repeat(func):
    """
    Decorator for repeating a function
    t can be either the object default (self.t),
    or specified in execution (t should not be a parameter of the decorated function)
    """
    restricted_func = argument_restrict(func)

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        t = kwargs.get('t', self.t)

        result = None
        for _ in range(t):
            result = restricted_func(self, *args, **kwargs)

        return result

    return argument_extend(Parameter('t', Parameter.KEYWORD_ONLY, default=None, annotation=int),
                           restrict=False)(wrapper)


class Assembly(object):
    """
    the main assembly object. according to our implementation, the main data of an assembly
    is his parents. we also keep a name for simulation puposes.
    """

    def __init__(self, parents: Iterable[Projectable], area_name: str, name: str, support_size: int, t: int = 1):
        self.parents: List[Projectable] = list(parents)
        self.area_name: str = area_name
        self.name: str = name
        self.support_size: int = support_size
        self.support: Dict[int, int] = {}

        """
        set default repeat amount (how many times to repeat each function) for
        this assembly object.
        """
        self.t = t

    def __repr__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)

    @staticmethod
    def fire_many(brain: Brain, projectables: Iterable[Projectable], area_name: str):
        """
        params:
        -the relevant brain object, as project is dependent on the brain instance.
        -a list of object which are projectable (Stimuli, Areas...) which will be projected
        -the target area's name.

        This function works by creating a "Parent tree", (Which is actually a directed acyclic graph) first,
        and then by going from the top layer, which will consist of stimuli, and traversing down the tree
        while firing each layer to areas its relevant descendant inhibit.

        For example, "firing" an assembly, we will first climb up its parent tree (Assemblies of course may have
        multiple parents, such as results of merge. Then we iterate over the resulting list in reverse, while firing
        each layer to the relevant areas, which are saved in a dictionary format:
        The thing we will project: areas to project it to
        """

        layers: List[Dict[Projectable, List[str]]] = [{stuff: [area_name] for stuff in projectables}]
        while any(isinstance(ass, Assembly) for ass in layers[-1]):
            prev_layer: Iterable[Assembly] = (ass for ass in layers[-1].keys() if not isinstance(ass, NamedStimulus))
            current_layer: Dict[Projectable, List[str]] = {}
            for ass in prev_layer:
                for parent in ass.parents:
                    current_layer[parent] = current_layer.get(ass, []) + [ass.area_name]

            layers.append(current_layer)

        layers = layers[::-1]
        for layer in layers:
            stimuli_mappings: Dict[str, List[str]] = {stim.name: areas
                                                      for stim, areas in
                                                      layer.items() if isinstance(stim, NamedStimulus)}

            area_mappings: Dict[str, List[str]] = {}
            for ass, areas in filter(lambda pair: isinstance(pair[0], Assembly), layer.items()):
                area_mappings[ass.area_name] = area_mappings.get(ass.area_name, []) + areas

            brain.project(stimuli_mappings, area_mappings)
            for ass in layer:
                if isinstance(ass, Assembly):
                    ass.update_support(brain.areas[ass.area_name].winners)

    def update_support(self, winners):
        oldest = 1
        for neuron in self.support:
            self.support[neuron] += 1
            oldest = max(oldest, self.support[neuron])
        for neuron in winners:
            self.support[neuron] = 1
        if len(self.support) <= self.support_size:
            return
        for neuron in self.support:
            if self.support[neuron] < oldest:
                continue
            del self.support[neuron]
    


    def fire(self, brain: Brain, mapping: Dict[Projectable, List[str]]):
        """
        fire for new API
        :param brain: the current brain; decided by the bindable decorator
        :param mapping: mapping of projectables into ares
        """
        for p in mapping:
            for t in mapping[p]:
                brain.inhibit(p, t)
        brain.next_round()
        for p in mapping:
            for t in mapping[p]:
                brain.disinhibit(p, t)





    @repeat
    def project(self, brain: Brain, area_name: str) -> 'Assembly':
        """
        a simple case of project many, with only one projectable parameter
        """
        projected_assembly: Assembly = Assembly([self], area_name, f"project({self.name}, {area_name})")
        Assembly.fire_many(brain, [self], area_name)
        projected_assembly.update_support(brain.areas[area_name].winners)
        return projected_assembly

    @repeat
    def reciprocal_project(self, brain: Brain, area_name: str) -> 'Assembly':
        projected_assembly: Assembly = self.project(brain, area_name)
        Assembly.fire_many(brain, [projected_assembly], self.area_name)
        self.update_support(brain.areas[self.area_name].winners)
        return projected_assembly

    @staticmethod
    @repeat
    def merge(brain: Brain, assembly1: 'Assembly', assembly2: 'Assembly', area_name: str) -> 'Assembly':
        """
        we create an "artificial" new assembly with x, y as parents, and then project_many
        to its area. this will create the effect of projecting stimultaneously, as described in the paper.
        """
        assert (assembly1.area_name != assembly2.area_name, "Areas are the same")
        merged_assembly: Assembly = Assembly([assembly1, assembly2], area_name,
                                             f"merge({assembly1.name}, {assembly2.name}, {area_name})")
        # TODO: Decide one of the two - Consult Edo Arad
        Assembly.fire_many(brain, [assembly1, assembly2], area_name)
        merged_assembly.update_support(brain.areas[area_name].winners)
        # OR: Assembly.fire_many(assembly1.brain, assembly1.parents + assembly2.parents)
        return merged_assembly

    @staticmethod
    @repeat
    def associate(brain: Brain, assembly1: 'Assembly', assembly2: 'Assembly'):
        assert (assembly1.area_name == assembly2.area_name, "Areas are not the same")
        Assembly.merge(brain, assembly1, assembly2, assembly1.area_name)

    @staticmethod
    def get_reads(brain: Brain, possible_assemblies: Iterable['Assembly'], area_name: str) -> Dict['Assembly', float]:
        """
        simulate the calculus up to a certain point on a copy of the original brain, and return the correlation
        of the parameters of the neurons after the effects of project with those of the original neurons.
        the purpose is to check how quickly an assembly stabilizes.
        ranks the list of assemblies with their correlations after the simulation.
        """
        original_area: Area = brain.areas[area_name]
        brain_copy: Brain = deepcopy(brain)
        brain_copy.disinhibit()  # TODO: Implement (Shani's & Adi Dinerstein's groups)

        assembly_reads: Dict['Assembly', float] = {}

        for ass in possible_assemblies:
            if ass.area_name != area_name:
                continue

            Assembly.fire_many(brain_copy, ass.parents, area_name)
            simulated_area = brain_copy.areas[area_name]
            simulated_ass_neurons = simulated_area.winners
            original_neurons = original_area.winners
            # TODO: Review scoring method, because assembly may "rotate"
            assembly_reads[ass] = len(set(simulated_ass_neurons).intersection(original_neurons)) / len(original_neurons)

        return assembly_reads

    @staticmethod
    def read(brain: Brain, possible_assemblies: Iterable['Assembly'], area_name: str) -> Optional['Assembly']:
        """
        simply return the most "stabilized" assembly, meaning the one with highest correlation.
        """
        area = brain.areas[area_name]
        for assembly in possible_assemblies:
            if set(area.winners).issubset(assembly.support.keys()):
                return assembly
        """
        Previous code:
        return max(Assembly.get_reads(brain, possible_assemblies, area_name))
        """
        return None

    """
    overriding arithmetic methods ( + , >> , <<) to make using common
    assembly operations easier.
    """

    """
    small class to represent many assemblies, to be fired with >> into an area.
    USAGE EXAMPLE: (ass1+ass2+ass3+ass4)>>"area name"
    """

    class AssemblyTuple(object):
        def __init__(self, lst):
            self.dat = lst

        def __add__(self, other):
            assert isinstance(other, Projectable.__args__)
            return Assembly.AssemblyTuple(self.dat + [other])

        def __rshift__(self, other):
            assert isinstance(other, str)
            Assembly.fire_many(self.dat[0].brain, self.dat, other)

    # ass1 < ass2 <=> ass1 is a child of ass2
    def __lt__(self, other):
        return other in self.parents

    # symmetric
    def __gt__(self, other):
        return self in other.parents

    # an assembly is defined by its area and parents.
    def __eq__(self, other):
        if not isinstance(other, Assembly): return False
        return set(self.parents) == set(other.parents) and self.area_name == other.area_name

    def __add__(self, other):
        return Assembly.AssemblyTuple([self, other])
